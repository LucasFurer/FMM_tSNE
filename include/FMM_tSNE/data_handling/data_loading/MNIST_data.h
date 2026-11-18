#pragma once

#include <cstdint>
#include <string>
#include <unsupported/Eigen/SparseExtra>
#include <vector>

namespace MNIST_data
{
std::vector<std::vector<float>> load_MNIST_images_fused();

std::vector<uint8_t> load_MNIST_labels_fused();

std::vector<std::vector<float>> load_MNIST_images(std::string& path);

std::vector<uint8_t> load_MNIST_labels(std::string& path);

uint32_t get_next_four_bytes(std::ifstream& in_strm);
}; // namespace MNIST_data
