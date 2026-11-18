#pragma once

#include <Eigen/Dense>
#include <FMM_tSNE/data_handling/data_loading/MNIST_data.h>
#include <vector>

namespace DataProcessing
{
std::vector<std::vector<float>> PCA(const std::vector<std::vector<float>>& input, int n_components);
} // namespace DataProcessing
