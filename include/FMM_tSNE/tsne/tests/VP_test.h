#pragma once

#include <FMM_tSNE/tsne/solvers/P_matrix/VP_tree/VP_tree.h>
#include <print>
#include <vector>

namespace VPTest
{
void find_NN_test()
{
    std::vector<std::vector<float>> points{
        {0, 0, 0},
        {-1, 2, 35},
        {2, 3, -12},
        {1.34, -0.23, 6.4},
        {32, 2390, 302},
    };

    VPTree VP_tree{};
    VP_tree.create_VP_tree(points);
    size_t NN = VP_tree.find_NN(points, 0);
    std::print("NN: {}", NN);
}
}; // namespace VPTest
