#pragma once

#include <FMM_tSNE/data_handling/data_entries/sparse_matrix.h>
#include <vector>

namespace DataProcessing
{
std::vector<std::vector<float>> PCA(const std::vector<std::vector<float>>& input, int n_components);

std::vector<std::vector<float>> PCA_sparse(const SparseMatrix& input, int n_components);
} // namespace DataProcessing
