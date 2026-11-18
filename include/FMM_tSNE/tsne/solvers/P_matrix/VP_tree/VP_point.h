#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

namespace VPPoint
{
inline float get_dist(const std::vector<float>& p1, const std::vector<float>& p2)
{
    float res{};
    for (size_t i{}; i < p1.size(); ++i)
    {
        res += (p1[i] - p2[i]) * (p1[i] - p2[i]);
    }
    return std::sqrt(res);
}

inline float get_median(std::vector<float>& vals)
{
    auto middle_upper = vals.begin() + vals.size() / 2;
    std::ranges::nth_element(vals, middle_upper);

    if (vals.size() % 2 == 1) return *middle_upper;

    auto middle_lower = std::ranges::max_element(vals.begin(), middle_upper);

    return (*middle_lower + *middle_upper) / 2.0f;
}
}; // namespace VPPoint
