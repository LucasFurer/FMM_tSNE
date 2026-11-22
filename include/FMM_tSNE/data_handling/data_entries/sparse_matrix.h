#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

struct SparseMatrix
{
    struct RowView
    {
        const uint32_t* idx;
        const float* val;
        size_t nnz;
    };

    size_t n_rows{};
    size_t n_cols{};

    std::vector<uint64_t> row_ptr; // n_rows + 1 entries
    std::vector<uint32_t> col_idx;
    std::vector<float> values;

    [[nodiscard]] size_t nnz() const { return values.size(); }

    [[nodiscard]] RowView row(size_t i) const
    {
        const auto b = static_cast<size_t>(row_ptr[i]);
        return {.idx = col_idx.data() + b, .val = values.data() + b, .nnz = static_cast<size_t>(row_ptr[i + 1]) - b};
    }

    // true iff every row has strictly increasing column indices, all < n_cols
    [[nodiscard]] bool rows_sorted() const
    {
        for (size_t i = 0; i < n_rows; ++i)
        {
            const RowView r = row(i);
            for (size_t e = 0; e < r.nnz; ++e)
            {
                if (r.idx[e] >= n_cols) return false;
                if (e > 0 && r.idx[e] <= r.idx[e - 1]) return false;
            }
        }
        return true;
    }

    // Sorts the entries of every row by column index.
    void sort_rows()
    {
        std::vector<std::pair<uint32_t, float>> tmp;
        for (size_t i = 0; i < n_rows; ++i)
        {
            const auto b = static_cast<size_t>(row_ptr[i]);
            const auto e = static_cast<size_t>(row_ptr[i + 1]);
            if (std::is_sorted(col_idx.begin() + static_cast<std::ptrdiff_t>(b),
                               col_idx.begin() + static_cast<std::ptrdiff_t>(e)))
                continue;
            tmp.clear();
            for (size_t j = b; j < e; ++j)
                tmp.emplace_back(col_idx[j], values[j]);
            std::sort(tmp.begin(), tmp.end(), [](const auto& a, const auto& b2) { return a.first < b2.first; });
            for (size_t j = b; j < e; ++j)
            {
                col_idx[j] = tmp[j - b].first;
                values[j] = tmp[j - b].second;
            }
        }
    }

    // Dense -> sparse (drops exact zeros). For small dense sets such as MNIST.
    // static SparseMatrix from_dense(const std::vector<std::vector<float>>& dense)
    // {
    //     SparseMatrix m;
    //     m.n_rows = dense.size();
    //     m.n_cols = dense.empty() ? 0 : dense[0].size();

    //     size_t count = 0;
    //     for (const auto& r : dense)
    //         for (const float v : r)
    //             count += (v != 0.0f);

    //     m.row_ptr.reserve(m.n_rows + 1);
    //     m.col_idx.reserve(count);
    //     m.values.reserve(count);
    //     m.row_ptr.push_back(0);
    //     for (const auto& r : dense)
    //     {
    //         for (size_t j = 0; j < r.size(); ++j)
    //         {
    //             if (r[j] == 0.0f) continue;
    //             m.col_idx.push_back(static_cast<uint32_t>(j));
    //             m.values.push_back(r[j]);
    //         }
    //         m.row_ptr.push_back(m.values.size());
    //     }
    //     return m;
    // }
};
