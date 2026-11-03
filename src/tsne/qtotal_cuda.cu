// clang-format off

#include <FMM_tSNE/tsne/Q_total_cuda.h>

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

// S threads cooperate on each sink point i (S = 1..32, power of two). Sources are streamed
// through shared memory in tiles; each lane takes every S-th source of the tile.
// Output: one partial sum per block (deterministic, no atomics).
template <int S>
__global__ void __launch_bounds__(BLOCK_SIZE)
qTotalKernel(const double2* __restrict__ pos, double* __restrict__ blockSums, int n)
{
    constexpr int SINKS_PER_BLOCK = BLOCK_SIZE / S;
    constexpr int PER_LANE = BLOCK_SIZE / S;

    __shared__ double2 tile[BLOCK_SIZE];
    __shared__ double red[BLOCK_SIZE];

    const int tid = threadIdx.x;
    const int lane = tid % S;
    const int i = blockIdx.x * SINKS_PER_BLOCK + tid / S;
    const bool valid = i < n;

    const double2 pi = valid ? pos[i] : make_double2(0.0, 0.0);
    double sum = 0.0;

    for (int base = 0; base < n; base += BLOCK_SIZE)
    {
        const int j = base + tid;
        tile[tid] = (j < n) ? pos[j] : make_double2(0.0, 0.0);
        __syncthreads();

        if (valid)
        {
            if (base + BLOCK_SIZE <= n)
            {
#pragma unroll 8
                for (int m = 0; m < PER_LANE; m++)
                {
                    const double2 q = tile[lane + m * S];
                    const double dx = q.x - pi.x;
                    const double dy = q.y - pi.y;
                    sum += 1.0 / (1.0 + (dx * dx + dy * dy));
                }
            }
            else
            {
                const int count = n - base;
                for (int k = lane; k < count; k += S)
                {
                    const double2 q = tile[k];
                    const double dx = q.x - pi.x;
                    const double dy = q.y - pi.y;
                    sum += 1.0 / (1.0 + (dx * dx + dy * dy));
                }
            }
        }
        __syncthreads();
    }

    // The loop visited j == i once, contributing exactly 1.0 (distance 0). Remove it (once per sink).
    if (valid && lane == 0)
        sum -= 1.0;

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
void launch(const double2* pos, double* partial, int n, int numBlocks)
{
    qTotalKernel<S><<<numBlocks, BLOCK_SIZE>>>(pos, partial, n);
}

// Roughly 2x the GPU's resident-thread capacity in flight.
int chooseThreadsPerSink(int n, int smCount)
{
    const long long target = static_cast<long long>(smCount) * 2048 * 2;
    int S = 1;
    while (S < 32 && static_cast<long long>(n) * S < target)
        S *= 2;
    return S;
}
}  // namespace

QTotalCuda::~QTotalCuda()
{
    release();
}

void QTotalCuda::release()
{
    if (d_pos) cudaFree(d_pos);
    if (d_partial) cudaFree(d_partial);
    d_pos = d_partial = nullptr;
    capacity = partialCapacity = 0;
}

void QTotalCuda::ensureCapacity(int n, int numBlocks)
{
    if (n > capacity)
    {
        if (d_pos) cudaFree(d_pos);
        d_pos = nullptr;
        capacity = 0;
        CUDA_CHECK(cudaMalloc(&d_pos, sizeof(double2) * static_cast<size_t>(n)));
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

double QTotalCuda::compute(const double* hostPos, int n)
{
    if (n <= 0)
        return 0.0;

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
    double* partial = static_cast<double*>(d_partial);

    switch (S)
    {
        case 1:  launch<1>(pos, partial, n, numBlocks); break;
        case 2:  launch<2>(pos, partial, n, numBlocks); break;
        case 4:  launch<4>(pos, partial, n, numBlocks); break;
        case 8:  launch<8>(pos, partial, n, numBlocks); break;
        case 16: launch<16>(pos, partial, n, numBlocks); break;
        default: launch<32>(pos, partial, n, numBlocks); break;
    }
    CUDA_CHECK(cudaGetLastError());

    std::vector<double> partialHost(numBlocks);
    CUDA_CHECK(cudaMemcpy(partialHost.data(), d_partial, sizeof(double) * numBlocks, cudaMemcpyDeviceToHost));

    double sum = 0.0;
    for (int b = 0; b < numBlocks; b++)
        sum += partialHost[b];
    return sum;
}
