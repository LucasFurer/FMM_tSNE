#include <FMM_tSNE/data_handling/data_processing/data_processing.h>

#include <Eigen/Dense>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <print>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

namespace DataProcessing
{
namespace
{
// Row-major so that "row j of V" (the l values belonging to gene j) is contiguous in memory.
using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Runs body(begin, end, thread_id) over [0, n) in chunks that are handed out dynamically.
// thread_id < n_threads, so each thread can own an accumulator without any locking.
template <class Body>
void parallel_chunks(size_t n, size_t n_threads, size_t chunk, Body&& body)
{
    if (n == 0) return;
    const size_t n_chunks = (n + chunk - 1) / chunk;
    n_threads = std::max<size_t>(1, std::min(n_threads, n_chunks));

    std::atomic<size_t> next{0};
    auto worker = [&](size_t tid)
    {
        for (;;)
        {
            const size_t b = next.fetch_add(chunk, std::memory_order_relaxed);
            if (b >= n) return;
            body(b, std::min(b + chunk, n), tid);
        }
    };

    std::vector<std::jthread> threads;
    threads.reserve(n_threads - 1);
    for (size_t t = 1; t < n_threads; ++t)
        threads.emplace_back(worker, t);
    worker(0);
}

// t[0..l) = x^T V for one sparse row x. V is row-major (n_cols x l).
inline void gather(const SparseMatrix::RowView& r, const double* V, size_t l, double* t)
{
    std::fill(t, t + l, 0.0);
    for (size_t e = 0; e < r.nnz; ++e)
    {
        const double v = r.val[e];
        const double* vr = V + static_cast<size_t>(r.idx[e]) * l;
        for (size_t c = 0; c < l; ++c)
            t[c] += v * vr[c];
    }
}

std::vector<double> column_means(const SparseMatrix& X, size_t T)
{
    const size_t g = X.n_cols;
    std::vector<std::vector<double>> acc(T);
    parallel_chunks(X.n_rows,
                    T,
                    4096,
                    [&](size_t begin, size_t end, size_t tid)
                    {
                        std::vector<double>& a = acc[tid];
                        if (a.empty()) a.assign(g, 0.0);
                        for (size_t e = X.row_ptr[begin]; e < X.row_ptr[end]; ++e)
                            a[X.col_idx[e]] += X.values[e];
                    });

    std::vector<double> mu(g, 0.0);
    for (const auto& a : acc)
        if (!a.empty())
            for (size_t j = 0; j < g; ++j)
                mu[j] += a[j];
    for (double& m : mu)
        m /= static_cast<double>(X.n_rows);
    return mu;
}

// W = Xc^T Xc V where Xc = X - 1 mu^T is never formed:  W = X^T (X V) - n mu (mu^T V).
// X^T (X V) is streamed over the rows of X.
RowMat apply_covariance(const SparseMatrix& X, const RowMat& V, const std::vector<double>& mu, size_t T)
{
    const size_t n = X.n_rows;
    const size_t g = X.n_cols;
    const size_t l = static_cast<size_t>(V.cols());

    std::vector<RowMat> acc(T);
    parallel_chunks(n,
                    T,
                    2048,
                    [&](size_t begin, size_t end, size_t tid)
                    {
                        RowMat& W = acc[tid];
                        if (W.size() == 0) W = RowMat::Zero(static_cast<Eigen::Index>(g), static_cast<Eigen::Index>(l));
                        std::vector<double> t(l);
                        for (size_t i = begin; i < end; ++i)
                        {
                            const auto r = X.row(i);
                            gather(r, V.data(), l, t.data());
                            for (size_t e = 0; e < r.nnz; ++e) // W += x t^T
                            {
                                const double v = r.val[e];
                                double* wr = W.data() + static_cast<size_t>(r.idx[e]) * l;
                                for (size_t c = 0; c < l; ++c)
                                    wr[c] += v * t[c];
                            }
                        }
                    });

    RowMat W = RowMat::Zero(static_cast<Eigen::Index>(g), static_cast<Eigen::Index>(l));
    for (const RowMat& a : acc)
        if (a.size() != 0) W += a;

    std::vector<double> c(l, 0.0); // c = mu^T V
    for (size_t j = 0; j < g; ++j)
        for (size_t cc = 0; cc < l; ++cc)
            c[cc] += mu[j] * V(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(cc));
    for (size_t j = 0; j < g; ++j)
        for (size_t cc = 0; cc < l; ++cc)
            W(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(cc)) -= static_cast<double>(n) * mu[j] * c[cc];
    return W;
}

// thin Q of a QR decomposition (g x l, orthonormal columns)
RowMat orthonormalize(const RowMat& A)
{
    const Eigen::Index g = A.rows();
    const Eigen::Index l = A.cols();
    Eigen::HouseholderQR<Eigen::MatrixXd> qr{Eigen::MatrixXd(A)};
    const Eigen::MatrixXd Q = qr.householderQ() * Eigen::MatrixXd::Identity(g, l);
    return RowMat(Q);
}
} // namespace
std::vector<std::vector<float>> PCA(const std::vector<std::vector<float>>& input, int n_components = 50)
{
    // std::print("begin PCA");
    if (input.empty() || input[0].empty()) throw std::invalid_argument("PCA: empty input");

    const Eigen::Index n = static_cast<Eigen::Index>(input.size());
    const Eigen::Index d = static_cast<Eigen::Index>(input[0].size());
    const Eigen::Index k = std::min<Eigen::Index>(n_components, std::min(n, d));

    // 1. Copy into an Eigen matrix (float to keep memory ~220 MB instead of ~440 MB)
    Eigen::MatrixXf X(n, d);
    for (Eigen::Index i = 0; i < n; ++i)
    {
        if (static_cast<Eigen::Index>(input[i].size()) != d)
            throw std::invalid_argument("PCA: rows have inconsistent sizes");
        X.row(i) = Eigen::Map<const Eigen::RowVectorXf>(input[i].data(), d);
    }

    // 2. Center the data
    Eigen::RowVectorXf mean = X.colwise().mean();
    X.rowwise() -= mean;

    // 3. SVD (only V is needed, so skip U to save memory and time)
    Eigen::BDCSVD<Eigen::MatrixXf> svd(X, Eigen::ComputeThinV);

    // 4. Principal axes = first k right singular vectors (d x k)
    Eigen::MatrixXf components = svd.matrixV().leftCols(k);

    // 5. Project: (n x d) * (d x k) = (n x k)
    Eigen::MatrixXf projected = X * components;

    // 6. Copy back to vector<vector<float>>
    std::vector<std::vector<float>> output(n, std::vector<float>(k));
    for (Eigen::Index i = 0; i < n; ++i)
        Eigen::Map<Eigen::RowVectorXf>(output[i].data(), k) = projected.row(i);

    // std::print("end PCA");
    return output;
}

// PCA of a sparse matrix (rows = points) with column centering. Neither a dense copy of `input`
// nor a genes x genes covariance matrix is ever formed; only the n x k result is dense.
//
// Randomized subspace iteration on C = Xc^T Xc with Xc = X - 1 mu^T kept implicit:
//     C V = X^T (X V) - n mu (mu^T V)
// X^T (X V) is streamed row by row, so extra memory is O(genes * k) per thread.
std::vector<std::vector<float>> PCA_sparse(const SparseMatrix& X, int n_components)
{
    // std::print("begin PCA_sparse");
    constexpr int kOversamples = 10;    // extra subspace dimensions (accuracy)
    constexpr int kPowerIterations = 4; // each one is a full pass over the data
    constexpr uint32_t kSeed = 42;

    if (n_components <= 0) throw std::invalid_argument("PCA_sparse: n_components must be positive");
    const size_t n = X.n_rows;
    const size_t g = X.n_cols;
    if (n < 2 || g == 0) throw std::invalid_argument("PCA_sparse: need at least 2 rows and 1 column");

    const size_t k = std::min<size_t>(static_cast<size_t>(n_components), std::min(n, g));
    const size_t l = std::min<size_t>(k + kOversamples, g);
    const size_t T = std::max<size_t>(1, std::thread::hardware_concurrency());

    const std::vector<double> mu = column_means(X, T);

    // ---- random orthonormal start, then subspace (power) iteration --------------------------
    std::mt19937_64 gen(kSeed);
    std::normal_distribution<double> normal(0.0, 1.0);
    RowMat G(static_cast<Eigen::Index>(g), static_cast<Eigen::Index>(l));
    for (Eigen::Index i = 0; i < G.size(); ++i)
        G.data()[i] = normal(gen);
    RowMat V = orthonormalize(G);
    for (int it = 0; it < kPowerIterations; ++it)
        V = orthonormalize(apply_covariance(X, V, mu, T));

    // ---- Rayleigh-Ritz: eigen-decompose the small l x l matrix V^T C V ----------------------
    // V^T C V = sum_i t_i t_i^T - n c c^T,   t_i = x_i^T V,   c = mu^T V
    std::vector<std::vector<double>> acc(T);
    parallel_chunks(n,
                    T,
                    2048,
                    [&](size_t begin, size_t end, size_t tid)
                    {
                        std::vector<double>& S = acc[tid];
                        if (S.empty()) S.assign(l * l, 0.0);
                        std::vector<double> t(l);
                        for (size_t i = begin; i < end; ++i)
                        {
                            gather(X.row(i), V.data(), l, t.data());
                            for (size_t a = 0; a < l; ++a)
                                for (size_t b = 0; b < l; ++b)
                                    S[a * l + b] += t[a] * t[b];
                        }
                    });

    Eigen::VectorXd c = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(l));
    for (size_t j = 0; j < g; ++j)
        for (size_t cc = 0; cc < l; ++cc)
            c(static_cast<Eigen::Index>(cc)) += mu[j] * V(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(cc));

    Eigen::MatrixXd Tm = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(l), static_cast<Eigen::Index>(l));
    for (const auto& S : acc)
        if (!S.empty())
            for (size_t a = 0; a < l; ++a)
                for (size_t b = 0; b < l; ++b)
                    Tm(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(b)) += S[a * l + b];
    Tm -= static_cast<double>(n) * c * c.transpose();

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(Tm); // eigenvalues in ascending order
    Eigen::MatrixXd U(static_cast<Eigen::Index>(l), static_cast<Eigen::Index>(k));
    for (size_t j = 0; j < k; ++j)
        U.col(static_cast<Eigen::Index>(j)) = eig.eigenvectors().col(static_cast<Eigen::Index>(l - 1 - j));

    const RowMat P = V * U; // principal axes, genes x k

    // ---- scores: (x_i - mu)^T P, written straight into the dense result ---------------------
    std::vector<double> shift(k, 0.0); // mu^T P
    for (size_t j = 0; j < g; ++j)
        for (size_t cc = 0; cc < k; ++cc)
            shift[cc] += mu[j] * P(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(cc));

    std::vector<std::vector<float>> output(n, std::vector<float>(k));
    parallel_chunks(n,
                    T,
                    2048,
                    [&](size_t begin, size_t end, size_t)
                    {
                        std::vector<double> t(k);
                        for (size_t i = begin; i < end; ++i)
                        {
                            gather(X.row(i), P.data(), k, t.data());
                            for (size_t c2 = 0; c2 < k; ++c2)
                                output[i][c2] = static_cast<float>(t[c2] - shift[c2]);
                        }
                    });
    // std::print("end PCA_sparse");
    return output;
}
} // namespace DataProcessing
