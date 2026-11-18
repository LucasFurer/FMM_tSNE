#pragma once

#include <cstdint>
#include <string>
#include <unsupported/Eigen/SparseExtra>
#include <vector>

namespace SparseDataLoader
{

std::vector<uint8_t> loadLabels(std::string path);

Eigen::SparseMatrix<double> loadPmatrix(std::string path);

}; // namespace SparseDataLoader
