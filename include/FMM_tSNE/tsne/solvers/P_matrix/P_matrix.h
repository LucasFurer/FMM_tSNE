#pragma once

#include <FMM_tSNE/timer/timer.h>
#include <FMM_tSNE/tsne/solvers/P_matrix/VP_tree/VP_tree.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <thread>
#include <utility>
#include <vector>

class PMatrix
{
  public:
    std::vector<unsigned int> m_row_P;
    std::vector<unsigned int> m_col_P;
    std::vector<double> m_val_P;

    std::vector<uint8_t> m_labels;

    // Number of worker threads. 0 = std::thread::hardware_concurrency().
    size_t m_n_threads{0};

    PMatrix() = default;

    PMatrix(const std::vector<std::vector<float>>& data, const std::vector<uint8_t>& labels, float perplexity)
        : m_labels(labels)
    {
        compute_P(data, perplexity);
    }

    void compute_P(const std::vector<std::vector<float>>& data, float perplexity)
    {
        const size_t n = data.size();

        m_row_P.assign(n + 1, 0);
        m_col_P.clear();
        m_val_P.clear();

        if (n < 2) return;

        const size_t k = std::min(static_cast<size_t>(3.0f * perplexity), n - 1);
        if (k == 0) return;

        const size_t T = m_n_threads != 0 ? m_n_threads : std::max<size_t>(1, std::thread::hardware_concurrency());

        Timer timer_NN{};
        // ---- 1. kNN + conditional probabilities, stored as n x k dense blocks -------------------
        // Row i occupies [i*k, (i+1)*k) in both arrays, sorted by neighbour index.
        // Every point is independent -> embarrassingly parallel.
        std::vector<uint32_t> nbr(n * k);
        std::vector<double> cond(n * k);

        {
            Timer timer_create_tree{};
            VPTree tree;
            tree.create_VP_tree(data);
            timer_create_tree.endTimer("create tree time");

            const double log_perp = std::log(static_cast<double>(perplexity));
            const std::vector<uint32_t>& order = tree.tree_order();

            parallel_for(n,
                         T,
                         64,
                         [&](size_t begin, size_t end)
                         {
                             // per-chunk scratch, so threads never share buffers
                             std::vector<uint32_t> ids;
                             std::vector<float> dists;
                             ids.reserve(k);
                             dists.reserve(k);
                             std::vector<double> d(k);
                             std::vector<double> p(k);
                             std::vector<std::pair<uint32_t, double>> row(k);

                             for (size_t idx = begin; idx < end; ++idx)
                             {
                                 const size_t i = order[idx]; // tree order = cache friendly

                                 // the tree already knows the distances; no need to recompute them
                                 tree.find_kNN(i, k, ids, dists);

                                 for (size_t j{}; j < k; ++j)
                                     d[j] = static_cast<double>(dists[j]) * static_cast<double>(dists[j]);

                                 conditional_row(d, p, log_perp);

                                 for (size_t j{}; j < k; ++j)
                                     row[j] = {ids[j], p[j]};
                                 std::sort(row.begin(),
                                           row.end(),
                                           [](const auto& a, const auto& b) { return a.first < b.first; });

                                 for (size_t j{}; j < k; ++j)
                                 {
                                     nbr[i * k + j] = row[j].first;
                                     cond[i * k + j] = row[j].second;
                                 }
                             }
                         });
        }
        timer_NN.endTimer("NN and conditional prob time");

        Timer timer_transpose{};
        // ---- 2. Transpose of the conditional matrix: parallel counting sort ---------------------
        // Rows are split into B contiguous blocks, one per thread. Each block counts its own
        // column occurrences (hist[j*B + b]); a prefix sum over (column, block) then gives every
        // block a private, disjoint output range inside each column. Blocks scatter without
        // locks, and because blocks are in row order every transposed row ends up sorted.
        const size_t B = std::min<size_t>(T, 16); // hist costs n*B*4 bytes, so cap it
        auto block_begin = [&](size_t b) { return b * n / B; };

        std::vector<uint32_t> hist(n * B, 0);
        parallel_for(B,
                     B,
                     1,
                     [&](size_t b, size_t)
                     {
                         for (size_t e = block_begin(b) * k; e < block_begin(b + 1) * k; ++e)
                             ++hist[static_cast<size_t>(nbr[e]) * B + b];
                     });

        std::vector<uint32_t> t_ptr(n + 1, 0);
        {
            uint32_t running = 0;
            for (size_t j{}; j < n; ++j)
            {
                t_ptr[j] = running;
                for (size_t b{}; b < B; ++b)
                {
                    const uint32_t c = hist[j * B + b];
                    hist[j * B + b] = running; // becomes block b's write cursor in column j
                    running += c;
                }
            }
            t_ptr[n] = running;
        }

        std::vector<uint32_t> t_col(n * k);
        std::vector<double> t_val(n * k);
        parallel_for(B,
                     B,
                     1,
                     [&](size_t b, size_t)
                     {
                         for (size_t i = block_begin(b); i < block_begin(b + 1); ++i)
                         {
                             for (size_t e{i * k}; e < (i + 1) * k; ++e)
                             {
                                 const uint32_t pos = hist[static_cast<size_t>(nbr[e]) * B + b]++;
                                 t_col[pos] = static_cast<uint32_t>(i);
                                 t_val[pos] = cond[e];
                             }
                         }
                     });
        std::vector<uint32_t>().swap(hist);
        timer_transpose.endTimer("transpose time");

        Timer timer_sym{};
        // ---- 3. Symmetrize: p_ij = (p_j|i + p_i|j) / (2n), by merging row i with row i of the transpose
        const double scale = 1.0 / (2.0 * static_cast<double>(n));

        auto merge_row = [&](size_t i, auto&& emit)
        {
            size_t a = i * k;
            const size_t a_end = (i + 1) * k;
            size_t b = t_ptr[i];
            const size_t b_end = t_ptr[i + 1];

            while (a < a_end && b < b_end)
            {
                if (nbr[a] < t_col[b])
                {
                    emit(nbr[a], cond[a] * scale);
                    ++a;
                }
                else if (nbr[a] > t_col[b])
                {
                    emit(t_col[b], t_val[b] * scale);
                    ++b;
                }
                else
                {
                    emit(nbr[a], (cond[a] + t_val[b]) * scale);
                    ++a;
                    ++b;
                }
            }
            for (; a < a_end; ++a)
                emit(nbr[a], cond[a] * scale);
            for (; b < b_end; ++b)
                emit(t_col[b], t_val[b] * scale);
        };

        // pass 1: row sizes (rows are independent)
        parallel_for(n,
                     T,
                     1024,
                     [&](size_t begin, size_t end)
                     {
                         for (size_t i = begin; i < end; ++i)
                         {
                             unsigned int count{};
                             merge_row(i, [&](uint32_t, double) { ++count; });
                             m_row_P[i + 1] = count;
                         }
                     });
        std::partial_sum(m_row_P.begin(), m_row_P.end(), m_row_P.begin());

        // pass 2: fill (each row writes into its own disjoint range)
        const size_t nnz = m_row_P[n];
        m_col_P.resize(nnz);
        m_val_P.resize(nnz);

        parallel_for(n,
                     T,
                     1024,
                     [&](size_t begin, size_t end)
                     {
                         for (size_t i = begin; i < end; ++i)
                         {
                             size_t out = m_row_P[i];
                             merge_row(i,
                                       [&](uint32_t col, double val)
                                       {
                                           m_col_P[out] = col;
                                           m_val_P[out] = val;
                                           ++out;
                                       });
                         }
                     });
        timer_sym.endTimer("symmetrize time");
    }

  private:
    // Runs body(begin, end) over [0, n) in chunks, on up to n_threads threads (the calling thread
    // included). Chunks are handed out dynamically through an atomic counter, so slow chunks don't
    // leave other threads idle. std::jthread joins automatically when `threads` goes out of scope.
    template <class Body>
    static void parallel_for(size_t n, size_t n_threads, size_t chunk, Body&& body)
    {
        if (n == 0) return;
        const size_t n_chunks = (n + chunk - 1) / chunk;
        n_threads = std::min(n_threads, n_chunks);
        if (n_threads <= 1)
        {
            for (size_t b = 0; b < n; b += chunk)
                body(b, std::min(b + chunk, n));
            return;
        }

        std::atomic<size_t> next{0};
        auto worker = [&]
        {
            for (;;)
            {
                const size_t b = next.fetch_add(chunk, std::memory_order_relaxed);
                if (b >= n) return;
                body(b, std::min(b + chunk, n));
            }
        };

        std::vector<std::jthread> threads;
        threads.reserve(n_threads - 1);
        for (size_t t = 1; t < n_threads; ++t)
            threads.emplace_back(worker);
        worker();
    }

    // Given squared distances `d` to the k nearest neighbours of one point, finds beta = 1 / (2 sigma^2)
    // such that the entropy of the conditional distribution equals log(perplexity) (in nats),
    // and writes the normalized conditional probabilities into `p`.
    //
    // Entropy increases with sigma, i.e. decreases with beta, so plain bisection works.
    static void conditional_row(std::vector<double>& d,
                                std::vector<double>& p,
                                const double log_perp,
                                const double tolerance = 1e-5,
                                const int max_iterations = 200)
    {
        const size_t k = d.size();

        // Shift by the smallest distance: cancels in the normalization, but guarantees
        // the nearest neighbour gets exp(0) = 1, so sum >= 1 and nothing underflows to 0/0.
        const double d_min = *std::min_element(d.begin(), d.end());
        double d_sum = 0.0;
        for (double& x : d)
        {
            x -= d_min;
            d_sum += x;
        }

        // Scale-aware starting point: beta ~ 1 / mean(d). Starting from a fixed 1.0 needs ~log2(scale)
        // extra halvings/doublings when distances are large (or tiny) before bisection even begins.
        const double mean_d = d_sum / static_cast<double>(k);
        double beta = mean_d > 0.0 ? 1.0 / mean_d : 1.0;
        double beta_min = 0.0;
        double beta_max = std::numeric_limits<double>::infinity();
        double sum = 1.0;

        for (int iter{}; iter < max_iterations; ++iter)
        {
            sum = 0.0;
            double weighted = 0.0;
            for (size_t j{}; j < k; ++j)
            {
                p[j] = std::exp(-beta * d[j]);
                sum += p[j];
                weighted += p[j] * d[j];
            }

            // H = -sum p log p, p = e / sum  =>  H = log(sum) + beta * weighted / sum
            const double diff = std::log(sum) + beta * weighted / sum - log_perp;

            if (std::abs(diff) < tolerance) break;

            if (diff > 0.0) // entropy too high -> Gaussian too wide -> increase beta
            {
                beta_min = beta;
                beta = std::isinf(beta_max) ? beta * 2.0 : (beta + beta_max) / 2.0;
            }
            else // entropy too low -> Gaussian too narrow -> decrease beta
            {
                beta_max = beta;
                beta = (beta_min == 0.0) ? beta / 2.0 : (beta + beta_min) / 2.0;
            }
        }

        // p and sum always belong to the same beta (the last one evaluated).
        const double inv = 1.0 / sum;
        for (size_t j{}; j < k; ++j)
            p[j] *= inv;
    }
};

// #pragma once
//
// #include <FMM_tSNE/timer/timer.h>
// #include <FMM_tSNE/tsne/solvers/P_matrix/VP_tree/VP_tree.h>
// #include <algorithm>
// #include <cmath>
// #include <cstddef>
// #include <cstdint>
// #include <limits>
// #include <numeric>
// #include <utility>
// #include <vector>
//
// class PMatrix
// {
//   public:
//     std::vector<unsigned int> m_row_P;
//     std::vector<unsigned int> m_col_P;
//     std::vector<double> m_val_P;
//
//     std::vector<uint8_t> m_labels;
//
//     PMatrix() = default;
//
//     PMatrix(const std::vector<std::vector<float>>& data, const std::vector<uint8_t>& labels, float perplexity)
//         : m_labels(labels)
//     {
//         compute_P(data, perplexity);
//     }
//
//     void compute_P(const std::vector<std::vector<float>>& data, float perplexity)
//     {
//         const size_t n = data.size();
//
//         m_row_P.assign(n + 1, 0);
//         m_col_P.clear();
//         m_val_P.clear();
//
//         if (n < 2) return;
//
//         const size_t k = std::min(static_cast<size_t>(3.0f * perplexity), n - 1);
//         if (k == 0) return;
//
//         Timer timer_NN{};
//         // ---- 1. kNN + conditional probabilities, stored as n x k dense blocks -------------------
//         // Row i occupies [i*k, (i+1)*k) in both arrays, sorted by neighbour index.
//         std::vector<uint32_t> nbr(n * k);
//         std::vector<double> cond(n * k);
//
//         {
//             Timer timer_create_tree{};
//             VPTree tree;
//             tree.create_VP_tree(data);
//             timer_create_tree.endTimer("create tree time");
//
//             const double log_perp = std::log(static_cast<double>(perplexity));
//             const auto sn = static_cast<std::ptrdiff_t>(n);
//
//             {
//                 std::vector<double> d(k);
//                 std::vector<double> p(k);
//                 std::vector<std::pair<uint32_t, double>> row(k);
//
//                 for (std::ptrdiff_t ii = 0; ii < sn; ++ii)
//                 {
//                     const auto i = static_cast<size_t>(ii);
//                     const std::vector<size_t> nn = tree.find_kNN(i, k);
//
//                     for (size_t j{}; j < k; ++j)
//                     {
//                         d[j] = static_cast<double>(sq_euclidean_dist(data[i], data[nn[j]]));
//                     }
//
//                     conditional_row(d, p, log_perp);
//
//                     for (size_t j{}; j < k; ++j)
//                     {
//                         row[j] = {static_cast<uint32_t>(nn[j]), p[j]};
//                     }
//                     std::sort(row.begin(), row.end(), [](const auto& a, const auto& b) { return a.first < b.first;
//                     });
//
//                     for (size_t j{}; j < k; ++j)
//                     {
//                         nbr[i * k + j] = row[j].first;
//                         cond[i * k + j] = row[j].second;
//                     }
//                 }
//             }
//         }
//         timer_NN.endTimer("NN and conditional prob time");
//
//         Timer timer_transpose{};
//         // ---- 2. Transpose of the conditional matrix (counting sort, O(nk)) ----------------------
//         // Iterating i in ascending order means every transposed row ends up sorted by column.
//         std::vector<uint32_t> t_ptr(n + 1, 0);
//         for (const uint32_t j : nbr)
//         {
//             ++t_ptr[j + 1];
//         }
//         std::partial_sum(t_ptr.begin(), t_ptr.end(), t_ptr.begin());
//
//         std::vector<uint32_t> t_col(n * k);
//         std::vector<double> t_val(n * k);
//         {
//             std::vector<uint32_t> pos(t_ptr.begin(), t_ptr.end() - 1);
//             for (size_t i{}; i < n; ++i)
//             {
//                 for (size_t e{i * k}; e < (i + 1) * k; ++e)
//                 {
//                     const uint32_t j = nbr[e];
//                     t_col[pos[j]] = static_cast<uint32_t>(i);
//                     t_val[pos[j]] = cond[e];
//                     ++pos[j];
//                 }
//             }
//         }
//         timer_transpose.endTimer("transpose time");
//
//         Timer timer_sym{};
//         // ---- 3. Symmetrize: p_ij = (p_j|i + p_i|j) / (2n), by merging row i with row i of the transpose
//         const double scale = 1.0 / (2.0 * static_cast<double>(n));
//         const auto sn = static_cast<std::ptrdiff_t>(n);
//
//         auto merge_row = [&](size_t i, auto&& emit)
//         {
//             size_t a = i * k;
//             const size_t a_end = (i + 1) * k;
//             size_t b = t_ptr[i];
//             const size_t b_end = t_ptr[i + 1];
//
//             while (a < a_end && b < b_end)
//             {
//                 if (nbr[a] < t_col[b])
//                 {
//                     emit(nbr[a], cond[a] * scale);
//                     ++a;
//                 }
//                 else if (nbr[a] > t_col[b])
//                 {
//                     emit(t_col[b], t_val[b] * scale);
//                     ++b;
//                 }
//                 else
//                 {
//                     emit(nbr[a], (cond[a] + t_val[b]) * scale);
//                     ++a;
//                     ++b;
//                 }
//             }
//             for (; a < a_end; ++a)
//                 emit(nbr[a], cond[a] * scale);
//             for (; b < b_end; ++b)
//                 emit(t_col[b], t_val[b] * scale);
//         };
//
//         // pass 1: row sizes
//         for (std::ptrdiff_t ii = 0; ii < sn; ++ii)
//         {
//             unsigned int count{};
//             merge_row(static_cast<size_t>(ii), [&](uint32_t, double) { ++count; });
//             m_row_P[static_cast<size_t>(ii) + 1] = count;
//         }
//         std::partial_sum(m_row_P.begin(), m_row_P.end(), m_row_P.begin());
//
//         // pass 2: fill
//         const size_t nnz = m_row_P[n];
//         m_col_P.resize(nnz);
//         m_val_P.resize(nnz);
//
//         for (std::ptrdiff_t ii = 0; ii < sn; ++ii)
//         {
//             size_t out = m_row_P[static_cast<size_t>(ii)];
//             merge_row(static_cast<size_t>(ii),
//                       [&](uint32_t col, double val)
//                       {
//                           m_col_P[out] = col;
//                           m_val_P[out] = val;
//                           ++out;
//                       });
//         }
//         timer_transpose.endTimer("symetrize time");
//     }
//
//   private:
//     // Given squared distances `d` to the k nearest neighbours of one point, finds beta = 1 / (2 sigma^2)
//     // such that the entropy of the conditional distribution equals log(perplexity) (in nats),
//     // and writes the normalized conditional probabilities into `p`.
//     //
//     // Entropy increases with sigma, i.e. decreases with beta, so plain bisection works.
//     static void conditional_row(std::vector<double>& d,
//                                 std::vector<double>& p,
//                                 const double log_perp,
//                                 const double tolerance = 1e-5,
//                                 const int max_iterations = 200)
//     {
//         const size_t k = d.size();
//
//         // Shift by the smallest distance: cancels in the normalization, but guarantees
//         // the nearest neighbour gets exp(0) = 1, so sum >= 1 and nothing underflows to 0/0.
//         const double d_min = *std::min_element(d.begin(), d.end());
//         for (double& x : d)
//             x -= d_min;
//
//         double beta = 1.0;
//         double beta_min = 0.0;
//         double beta_max = std::numeric_limits<double>::infinity();
//         double sum = 1.0;
//
//         for (int iter{}; iter < max_iterations; ++iter)
//         {
//             sum = 0.0;
//             double weighted = 0.0;
//             for (size_t j{}; j < k; ++j)
//             {
//                 p[j] = std::exp(-beta * d[j]);
//                 sum += p[j];
//                 weighted += p[j] * d[j];
//             }
//
//             // H = -sum p log p, p = e / sum  =>  H = log(sum) + beta * weighted / sum
//             const double diff = std::log(sum) + beta * weighted / sum - log_perp;
//
//             if (std::abs(diff) < tolerance) break;
//
//             if (diff > 0.0) // entropy too high -> Gaussian too wide -> increase beta
//             {
//                 beta_min = beta;
//                 beta = std::isinf(beta_max) ? beta * 2.0 : (beta + beta_max) / 2.0;
//             }
//             else // entropy too low -> Gaussian too narrow -> decrease beta
//             {
//                 beta_max = beta;
//                 beta = (beta_min == 0.0) ? beta / 2.0 : (beta + beta_min) / 2.0;
//             }
//         }
//
//         // p and sum always belong to the same beta (the last one evaluated).
//         const double inv = 1.0 / sum;
//         for (size_t j{}; j < k; ++j)
//             p[j] *= inv;
//     }
//
//     static float sq_euclidean_dist(const std::vector<float>& p1, const std::vector<float>& p2)
//     {
//         float squared_sum{};
//         for (size_t i{}; i < p1.size(); ++i)
//         {
//             const float diff = p1[i] - p2[i];
//             squared_sum += diff * diff;
//         }
//         return squared_sum;
//     }
// };
