#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <vector>

class PMatrix
{
  public:
    std::vector<unsigned int> m_row_P;
    std::vector<unsigned int> m_col_P;
    std::vector<double> m_val_P;

    std::vector<uint8_t> m_labels;

    std::vector<std::vector<float>> P;
    std::vector<float> sigma;

    PMatrix() = default;

    PMatrix(const std::vector<std::vector<float>>& data, const std::vector<uint8_t>& labels, float perplexity)
    {
        std::vector<uint8_t> data_labels_1k;
        std::ranges::copy(labels.begin(), labels.begin() + 1000, std::back_inserter(data_labels_1k));
        m_labels = data_labels_1k;

        std::vector<std::vector<float>> data_images_1k_reduced;
        std::ranges::copy(data.begin(), data.begin() + 1000, std::back_inserter(data_images_1k_reduced));

        P = compute_P(data_images_1k_reduced, perplexity);

        make_P_sparse();
    }

    void make_P_sparse()
    {
        const size_t n = P.size();

        m_col_P.clear();
        m_row_P.clear();
        m_val_P.clear();

        // col_P : column pointers (size n+1)
        // row_P : row indices     (size nnz)
        m_row_P.reserve(n + 1);

        for (size_t col{}; col < n; ++col)
        {
            m_row_P.push_back(static_cast<unsigned int>(m_val_P.size())); // start of this column

            for (size_t row{}; row < n; ++row)
            {
                if (P[row][col] != 0.0f)
                {
                    m_col_P.push_back(static_cast<unsigned int>(row));
                    m_val_P.push_back(P[row][col]);
                }
            }
        }
        m_row_P.push_back(static_cast<unsigned int>(m_val_P.size())); // final entry = nnz
    }

    std::vector<std::vector<float>> compute_P(const std::vector<std::vector<float>>& data, float perplexity)
    {
        const auto n{data.size()};

        const std::vector<std::vector<float>> sq_dist = sq_dist_matrix(data);
        const std::vector<std::vector<float>> P_cond = compute_conditional_P(sq_dist, perplexity);

        // const std::vector<float> normalization = compute_normalization(sq_dist);

        std::vector<std::vector<float>> result(n, std::vector<float>(n));
        for (size_t i{}; i < n; ++i)
        {
            for (size_t j{i + 1}; j < n; ++j)
            {
                result[i][j] = (P_cond[i][j] + P_cond[j][i]) / (2.0f * static_cast<float>(n));
                result[j][i] = result[i][j];
            }
        }

        return result;
    }

    // For every point i, binary-searches sigma_i so that Perp(P_i) = perplexity.
    // The search is done over beta = 1 / (2 sigma_i^2); perplexity increases monotonically
    // with sigma_i, so it decreases monotonically with beta.
    std::vector<std::vector<float>> compute_conditional_P(const std::vector<std::vector<float>>& sq_dist,
                                                          float perplexity,
                                                          double tolerance = 1e-5,
                                                          int max_iterations = 200)
    {
        const auto n{sq_dist.size()};
        const double target_entropy = std::log2(static_cast<double>(perplexity)); // in bits

        std::vector<std::vector<float>> P_cond(n, std::vector<float>(n));
        sigma.assign(n, 0.0f);

        std::vector<double> row(n);

        for (size_t i{}; i < n; ++i)
        {
            // Shift distances by the row minimum (excluding i) for numerical stability.
            // The shift cancels in the normalization, but guarantees the nearest neighbor gets exp(0) = 1.
            double d_min = std::numeric_limits<double>::infinity();
            for (size_t j{}; j < n; ++j)
            {
                if (j != i) d_min = std::min(d_min, static_cast<double>(sq_dist[i][j]));
            }

            double beta = 1.0;
            double beta_min = 0.0;
            double beta_max = std::numeric_limits<double>::infinity();

            for (int iter{}; iter < max_iterations; ++iter)
            {
                const double entropy = evaluate(beta, n, i, row, sq_dist, d_min).second;
                const double diff = entropy - target_entropy;

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

            // Final row at the beta we ended with.
            const double sum = evaluate(beta, n, i, row, sq_dist, d_min).first;
            for (size_t j{}; j < n; ++j)
            {
                if (j != i) P_cond[i][j] = static_cast<float>(row[j] / sum);
            }
            sigma[i] = static_cast<float>(std::sqrt(1.0 / (2.0 * beta)));
        }

        return P_cond;
    }

    // Fills `row` with unnormalized exp(-beta * d'), returns {sum, entropy in bits}.
    std::pair<double, double> evaluate(const double beta,
                                       const size_t n,
                                       const size_t i,
                                       std::vector<double>& row,
                                       const std::vector<std::vector<float>>& sq_dist,
                                       const double d_min)
    {
        double sum{};
        double weighted{};
        for (size_t j{}; j < n; ++j)
        {
            if (j == i) continue;
            const double d = static_cast<double>(sq_dist[i][j]) - d_min;
            row[j] = std::exp(-beta * d);
            sum += row[j];
            weighted += row[j] * d;
        }
        // H = -sum p log p with p = row / sum  =>  H = log(sum) + beta * weighted / sum (nats)
        const double entropy_bits = (std::log(sum) + beta * weighted / sum) / std::numbers::ln2;
        return std::pair{sum, entropy_bits};
    };

    // std::vector<float> compute_normalization(const std::vector<std::vector<float>>& sq_dist)
    // {
    //     const auto n{sq_dist.size()};

    //     std::vector<float> normalization(n);
    //     for (size_t i{}; i < n; ++i)
    //     {
    //         for (size_t j{}; j < n; ++j)
    //         {
    //             if (i == j) continue;
    //             normalization[i] += minus_exp_d_ij_sq(i, j, sq_dist);
    //         }
    //     }
    //     return normalization;
    // }

    // float compute_unsymmetric_P(size_t i,
    //                             size_t j,
    //                             const std::vector<std::vector<float>>& sq_dist,
    //                             const std::vector<float>& precomputed_exp_minus_d)
    // {
    //     return minus_exp_d_ij_sq(i, j, sq_dist) / precomputed_exp_minus_d[i];
    // }

    // float minus_exp_d_ij_sq(size_t i, size_t j, const std::vector<std::vector<float>>& sq_dist)
    // {
    //     float sigma_i = 0.5f;

    //     return std::exp(-sq_dist[i][j] / (2.0f * sigma_i * sigma_i));
    // }

    std::vector<std::vector<float>> sq_dist_matrix(const std::vector<std::vector<float>>& data)
    {
        const auto n{data.size()};
        std::vector<std::vector<float>> sq_dist(n, std::vector<float>(n));

        for (size_t i{}; i < n; ++i)
        {
            for (size_t j{i + 1}; j < n; ++j)
            {
                sq_dist[i][j] = sq_euclidean_dist(data[i], data[j]);
                sq_dist[j][i] = sq_dist[i][j];
            }
        }

        return sq_dist;
    }

    float sq_euclidean_dist(const std::vector<float>& p1, const std::vector<float>& p2)
    {
        float squared_sum{};
        for (size_t i{}; i < p1.size(); ++i)
        {
            squared_sum += (p1[i] - p2[i]) * (p1[i] - p2[i]);
        }
        return squared_sum;
    }
};
