#pragma once

// ---------------------------------------------------------------------------
// GPU computation of the t-SNE normalisation term
//     QijTotal = sum_{i != j} 1 / (1 + |p_i - p_j|^2)
// Works on raw interleaved doubles (x0,y0,x1,y1,...) so this header has no glm / CUDA
// dependencies and can be included from plain C++ files. Implemented in qtotal_cuda.cu.
// ---------------------------------------------------------------------------
class QTotalCuda
{
  public:
    QTotalCuda() = default;
    ~QTotalCuda();

    QTotalCuda(const QTotalCuda&) = delete;
    QTotalCuda& operator=(const QTotalCuda&) = delete;

    // hostPos: n * 2 doubles. Returns QijTotal (double precision).
    double compute(const double* hostPos, int n);

  private:
    void ensureCapacity(int n, int numBlocks);
    void release();

    void* d_pos = nullptr;     // double2[capacity]
    void* d_partial = nullptr; // double[numBlocks]
    int capacity = 0;
    int partialCapacity = 0;
    int smCount = 0;
};
