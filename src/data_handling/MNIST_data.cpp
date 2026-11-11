#include <FMM_tSNE/data_handling/MNIST_data.h>

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

namespace MNIST_data
{

std::vector<std::vector<float>> load_MNIST_images_fused()
{
    std::string path_images_60k = "TSNE_data_input/train-images-idx3-ubyte";
    auto data_images_60k = MNIST_data::load_MNIST_images(path_images_60k);

    std::string path_images_10k = "TSNE_data_input/t10k-images-idx3-ubyte";
    auto data_images_10k = MNIST_data::load_MNIST_images(path_images_10k);

    data_images_60k.insert(data_images_60k.end(), data_images_10k.begin(), data_images_10k.end());

    return data_images_60k;
}

std::vector<uint8_t> load_MNIST_labels_fused()
{

    std::string path_labels_60k = "TSNE_data_input/train-labels-idx1-ubyte";
    auto data_labels_60k = MNIST_data::load_MNIST_labels(path_labels_60k);

    std::string path_labels_10k = "TSNE_data_input/t10k-labels-idx1-ubyte";
    auto data_labels_10k = MNIST_data::load_MNIST_labels(path_labels_10k);

    data_labels_60k.insert(data_labels_60k.end(), data_labels_10k.begin(), data_labels_10k.end());

    return data_labels_60k;
}

std::vector<std::vector<float>> load_MNIST_images(std::string& path)
{
    std::ifstream in_strm(path);
    if (!in_strm)
    {
        std::cerr << "failed to open: " << path << "\n";
        return {};
    }

    uint32_t magicNumber = get_next_four_bytes(in_strm);
    uint32_t amount = get_next_four_bytes(in_strm);
    uint32_t rows = get_next_four_bytes(in_strm);
    uint32_t cols = get_next_four_bytes(in_strm);
    std::print("magicNumber: {}\namount: {}\nrows: {}\ncols: {}\n", magicNumber, amount, rows, cols);

    size_t entry_byte_size = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    std::vector<std::vector<float>> result(amount, std::vector<float>(entry_byte_size, 0.0));

    for (size_t entry{}; entry < amount; ++entry)
    {
        std::vector<uint8_t> char_buffer(entry_byte_size);
        in_strm.read(reinterpret_cast<char*>(char_buffer.data()), static_cast<std::streamsize>(entry_byte_size));

        for (size_t c{}; c < entry_byte_size; ++c)
            result[entry][c] = static_cast<float>(char_buffer[c]);
    }

    return result;
}

std::vector<uint8_t> load_MNIST_labels(std::string& path)
{
    std::ifstream in_strm(path);
    if (!in_strm)
    {
        std::cerr << "failed to open: " << path << "\n";
        return {};
    }

    uint32_t magicNumber = get_next_four_bytes(in_strm);
    uint32_t amount = get_next_four_bytes(in_strm);
    std::print("magicNumber: {}\namount: {}\n", magicNumber, amount);

    std::vector<uint8_t> result(amount);

    in_strm.read(reinterpret_cast<char*>(result.data()), amount);

    return result;
}

uint32_t get_next_four_bytes(std::ifstream& in_strm)
{
    std::array<uint8_t, 4> four_bytes;
    in_strm.read(reinterpret_cast<char*>(four_bytes.data()), 4);

    return (static_cast<uint32_t>(four_bytes[0]) << 24 | static_cast<uint32_t>(four_bytes[1]) << 16 |
            static_cast<uint32_t>(four_bytes[2]) << 8 | static_cast<uint32_t>(four_bytes[3]));
}

}; // namespace MNIST_data
