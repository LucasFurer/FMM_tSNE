#pragma once

// ---------------------------------------------------------------------------
// Non-template CUDA backend (implemented in nbody_solver_naive_cuda.cu).
// Works on raw interleaved double arrays so this header can be included from
// plain C++ translation units (no CUDA headers needed).
// ---------------------------------------------------------------------------
class NaiveCudaBackend
{
  public:
    NaiveCudaBackend() = default;
    ~NaiveCudaBackend();

    NaiveCudaBackend(const NaiveCudaBackend&) = delete;
    NaiveCudaBackend& operator=(const NaiveCudaBackend&) = delete;

    // hostPos   : n * 2 doubles (x0,y0,x1,y1,...)
    // hostDeriv : n * 2 doubles, OUTPUT. Contribution of all sources to each sink (not accumulated
    //             onto previous values; the caller adds it).
    // total     : OUTPUT, sum over all i != j of 1 / (1 + |p_i - p_j|^2)
    void solve(const double* hostPos, double* hostDeriv, int n, double& total);

  private:
    void ensureCapacity(int n, int numBlocks);
    void release();

    void* d_pos = nullptr;     // double2[capacity]
    void* d_deriv = nullptr;   // double2[capacity]
    void* d_partial = nullptr; // double[numBlocks]
    int capacity = 0;
    int partialCapacity = 0;
    int smCount = 0; // cached multiprocessor count of the active device
};
