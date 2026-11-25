#pragma once

#include <string>
#include <vector>

namespace StarData
{
std::vector<std::vector<float>> load_full_star_data(size_t row_limit,
                                                    const std::string& path = "TSNE_data_input/star/");
}
