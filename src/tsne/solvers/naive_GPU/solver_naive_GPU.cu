// clang-format off

#include <FMM_tSNE/tsne/solvers/naive_GPU/solver_naive_GPU_backend.h>

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                                           \
    do                                                                                             \
    {                                                                                              \
        cudaError_t err__ = (call);                                                                \
        if (err__ != cudaSuccess)                                                                  \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err__) +     \
                                     " at " + __FILE__ + ":" + std::to_string(__LINE__));          \
    } while (0)

namespace
{
constexpr int BLOCK_SIZE = 256;  // power of two, multiple of 32

// S = number of threads cooperating on one sink point (power of two, 1..32).
// A block handles BLOCK_SIZE / S sinks. Sources are streamed through shared memory in
// tiles of BLOCK_SIZE; within a tile, the S lanes of a sink take every S-th source.
// Lanes of one sink always live in the same warp, so they are combined with shuffles.
template <int S>
__global__ void __launch_bounds__(BLOCK_SIZE)
naiveNbodyKernel(const double2* __restrict__ pos,
                 double2* __restrict__ deriv,
                 double* __restrict__ blockSums,
                 int n)
{
    constexpr int SINKS_PER_BLOCK = BLOCK_SIZE / S;
    constexpr int PER_LANE = BLOCK_SIZE / S;  // sources per lane in a full tile

    __shared__ double2 tile[BLOCK_SIZE];
    __shared__ double red[BLOCK_SIZE];

    const int tid = threadIdx.x;
    const int lane = tid % S;
    const int i = blockIdx.x * SINKS_PER_BLOCK + tid / S;
    const bool valid = i < n;

    const double2 pi = valid ? pos[i] : make_double2(0.0, 0.0);

    double sum = 0.0;
    double dx = 0.0;
    double dy = 0.0;

    for (int base = 0; base < n; base += BLOCK_SIZE)
    {
        const int j = base + tid;
        tile[tid] = (j < n) ? pos[j] : make_double2(0.0, 0.0);
        __syncthreads();

        if (valid)
        {
            if (base + BLOCK_SIZE <= n)
            {
                // Full tile: compile-time trip count so the loop can be unrolled.
#pragma unroll 8
                for (int m = 0; m < PER_LANE; m++)
                {
                    const double2 q = tile[lane + m * S];
                    const double diffx = pi.x - q.x;
                    const double diffy = pi.y - q.y;
                    const double d2 = diffx * diffx + diffy * diffy;  // == |diff|^2, no sqrt
                    const double f = 1.0 / (1.0 + d2);
                    sum += f;
                    const double ff = f * f;
                    dx += ff * diffx;
                    dy += ff * diffy;
                }
            }
            else
            {
                const int count = n - base;
                for (int k = lane; k < count; k += S)
                {
                    const double2 q = tile[k];
                    const double diffx = pi.x - q.x;
                    const double diffy = pi.y - q.y;
                    const double d2 = diffx * diffx + diffy * diffy;
                    const double f = 1.0 / (1.0 + d2);
                    sum += f;
                    const double ff = f * f;
                    dx += ff * diffx;
                    dy += ff * diffy;
                }
            }
        }
        __syncthreads();
    }

    // The loop above also visited j == i once. That pair has diff == 0, so it adds exactly 0 to
    // the derivative and exactly 1.0 to the normalisation sum; remove that 1.0 (once per sink).
    if (valid && lane == 0)
        sum -= 1.0;

    // Combine the S lanes of each sink (all threads of the warp take part in the shuffles).
#pragma unroll
    for (int off = S / 2; off > 0; off >>= 1)
    {
        dx += __shfl_xor_sync(0xffffffffu, dx, off);
        dy += __shfl_xor_sync(0xffffffffu, dy, off);
    }
    if (valid && lane == 0)
        deriv[i] = make_double2(dx, dy);

    // Deterministic block reduction of the normalisation term (no atomics).
    red[tid] = sum;
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            red[tid] += red[tid + s];
        __syncthreads();
    }
    if (tid == 0)
        blockSums[blockIdx.x] = red[0];
}

template <int S>
void launch(const double2* pos, double2* deriv, double* partial, int n, int numBlocks)
{
    naiveNbodyKernel<S><<<numBlocks, BLOCK_SIZE>>>(pos, deriv, partial, n);
}

// Pick threads-per-sink so that roughly 2x the GPU's resident-thread capacity is in flight.
int chooseThreadsPerSink(int n, int smCount)
{
    const long long target = static_cast<long long>(smCount) * 2048 * 2;
    int S = 1;
    while (S < 32 && static_cast<long long>(n) * S < target)
        S *= 2;
    return S;
}
}  // namespace

NaiveCudaBackend::~NaiveCudaBackend()
{
    release();
}

void NaiveCudaBackend::release()
{
    if (d_pos) cudaFree(d_pos);
    if (d_deriv) cudaFree(d_deriv);
    if (d_partial) cudaFree(d_partial);
    d_pos = d_deriv = d_partial = nullptr;
    capacity = partialCapacity = 0;
}

void NaiveCudaBackend::ensureCapacity(int n, int numBlocks)
{
    if (n > capacity)
    {
        if (d_pos) cudaFree(d_pos);
        if (d_deriv) cudaFree(d_deriv);
        d_pos = d_deriv = nullptr;
        capacity = 0;
        CUDA_CHECK(cudaMalloc(&d_pos, sizeof(double2) * static_cast<size_t>(n)));
        CUDA_CHECK(cudaMalloc(&d_deriv, sizeof(double2) * static_cast<size_t>(n)));
        capacity = n;
    }
    if (numBlocks > partialCapacity)
    {
        if (d_partial) cudaFree(d_partial);
        d_partial = nullptr;
        partialCapacity = 0;
        CUDA_CHECK(cudaMalloc(&d_partial, sizeof(double) * static_cast<size_t>(numBlocks)));
        partialCapacity = numBlocks;
    }
}

void NaiveCudaBackend::solve(const double* hostPos, double* hostDeriv, int n, double& total)
{
    total = 0.0;
    if (n <= 0)
        return;

    if (smCount == 0)
    {
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, dev));
    }

    const int S = chooseThreadsPerSink(n, smCount);
    const int sinksPerBlock = BLOCK_SIZE / S;
    const int numBlocks = (n + sinksPerBlock - 1) / sinksPerBlock;

    ensureCapacity(n, numBlocks);

    CUDA_CHECK(cudaMemcpy(d_pos, hostPos, sizeof(double2) * static_cast<size_t>(n), cudaMemcpyHostToDevice));

    const double2* pos = static_cast<const double2*>(d_pos);
    double2* deriv = static_cast<double2*>(d_deriv);
    double* partial = static_cast<double*>(d_partial);

    switch (S)
    {
        case 1:  launch<1>(pos, deriv, partial, n, numBlocks); break;
        case 2:  launch<2>(pos, deriv, partial, n, numBlocks); break;
        case 4:  launch<4>(pos, deriv, partial, n, numBlocks); break;
        case 8:  launch<8>(pos, deriv, partial, n, numBlocks); break;
        case 16: launch<16>(pos, deriv, partial, n, numBlocks); break;
        default: launch<32>(pos, deriv, partial, n, numBlocks); break;
    }
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(hostDeriv, d_deriv, sizeof(double2) * static_cast<size_t>(n), cudaMemcpyDeviceToHost));

    std::vector<double> partialHost(numBlocks);
    CUDA_CHECK(cudaMemcpy(partialHost.data(), d_partial, sizeof(double) * numBlocks, cudaMemcpyDeviceToHost));

    double sum = 0.0;
    for (int b = 0; b < numBlocks; b++)
        sum += partialHost[b];
    total = sum;
}
