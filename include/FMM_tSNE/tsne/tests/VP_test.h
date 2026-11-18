#pragma once

#include <FMM_tSNE/tsne/solvers/P_matrix/VP_tree/VP_point.h>
#include <FMM_tSNE/tsne/solvers/P_matrix/VP_tree/VP_tree.h>
#include <limits>
#include <print>
#include <vector>

namespace VPTest
{
void find_NN_test()
{
    auto find_NN_bf = [](const std::vector<std::vector<float>>& points, size_t q) -> size_t
    {
        size_t closest = q;
        float smallest_dist = std::numeric_limits<float>::infinity();
        for (size_t p{}; p < points.size(); ++p)
        {
            if (p != q)
            {
                float dist = VPPoint::get_dist(points[q], points[p]);
                if (dist < smallest_dist)
                {
                    closest = p;
                    smallest_dist = dist;
                }
            }
        }
        return closest;
    };

    std::vector<std::vector<float>> points{
        {0, 0, 0},
        {-1, 2, 35},
        {2, 3, -12},
        {1.34, -0.23, 6.4},
        {32, 2390, 302},
        {1, 20, 302},
        {32, 39, 302},
        {5, 2, 2},
        {2, 9, 3},
    };

    size_t NN_correct = find_NN_bf(points, 0);

    VPTree VP_tree{};
    VP_tree.create_VP_tree(points);
    std::vector<size_t> NNs = VP_tree.find_kNN(points, 0, 1);

    std::print("NN_correct: {}, NN_VP: {}\n", NN_correct, NNs[0]);
}
}; // namespace VPTest
