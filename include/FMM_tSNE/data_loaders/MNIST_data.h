#pragma once

#include <string>
#include <unsupported/Eigen/SparseExtra>
#include <vector>

namespace MNIST_data
{

float*
loadMNIST(unsigned int* dataAmount, unsigned int* dataDimension, const char* path, int maxAmount);

std::vector<uint8_t> loadLabels(std::string path);

Eigen::SparseMatrix<double> loadPmatrix(std::string path);

uint32_t getNextFourBytes(FILE* file);

}; // namespace MNIST_data
