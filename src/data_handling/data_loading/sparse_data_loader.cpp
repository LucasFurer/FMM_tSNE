#include <FMM_tSNE/data_handling/data_loading/sparse_data_loader.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <iostream>
#include <iterator>
#include <print>
#include <string>
#include <unsupported/Eigen/SparseExtra>
#include <vector>

namespace SparseDataLoader
{

std::vector<uint8_t> loadLabels(std::string path)
{
    std::ifstream file(path, std::ios::binary);

    if (!file)
    {
        std::cerr << "Error opening file!" << std::endl;
    }

    return std::vector<uint8_t>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

Eigen::SparseMatrix<double> loadPmatrix(std::string path)
{
    Eigen::SparseMatrix<double> Pmatrix;

    std::ifstream file(path);
    if (file.is_open())
    {
        Eigen::loadMarket(Pmatrix, path);
        std::cout << "Matrix loaded successfully!" << std::endl;
        // Pmatrix.coeff(i, j)
        // Pmatrix.rows()
        // Pmatrix.cols()
    }
    else
    {
        std::cerr << "Failed to open " + (std::string) path + " file!" << std::endl;
    }

    return Pmatrix;
}

}; // namespace SparseDataLoader
