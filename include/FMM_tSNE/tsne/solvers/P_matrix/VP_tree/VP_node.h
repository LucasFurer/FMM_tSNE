// #pragma once
//
// #include <FMM_tSNE/tsne/solvers/P_matrix/VP_tree/VP_point.h>
// #include <algorithm>
// #include <cstddef>
// #include <random>
// #include <vector>
//
// struct VPNode
// {
//     size_t p{};
//     float l_lower_bound{};
//     float l_upper_bound{};
//     float r_lower_bound{};
//     float r_upper_bound{};
//     // float mu{};
//
//     size_t node_l{};
//     size_t node_r{};
//
//     void set_vp(const std::vector<std::vector<float>>& points, std::mt19937& gen, std::vector<size_t>&
//     contained_points)
//     {
//         size_t max_sample_count = std::min(size_t{32}, contained_points.size());
//
//         std::uniform_int_distribution<size_t> dis(0u, contained_points.size() - 1u);
//
//         // select random points to test as vantage point
//         std::vector<size_t> VP_rand_samples;
//         VP_rand_samples.reserve(max_sample_count);
//         for (size_t vp = 0; vp < max_sample_count; ++vp)
//             VP_rand_samples.push_back(dis(gen));
//
//         // keep track of best point
//         float best_spread{-1.0f};
//         size_t best_vp;
//
//         std::vector<float> distances;
//         distances.reserve(max_sample_count);
//         for (size_t vp = 0; vp < max_sample_count; ++vp)
//         {
//             const std::vector<float>& VP_point = points[contained_points[VP_rand_samples[vp]]];
//
//             distances.clear();
//             for (size_t eval_p = 0; eval_p < max_sample_count; ++eval_p)
//             {
//                 const std::vector<float>& eval_point = points[contained_points[VP_rand_samples[eval_p]]];
//
//                 distances.push_back(VPPoint::get_dist(VP_point, eval_point));
//             }
//             float median = VPPoint::get_median(distances);
//             float spread{};
//             for (float d : distances)
//             {
//                 spread += (d - median) * (d - median);
//             }
//
//             // I think this line is actually useless since we only care about who is the largest
//             // spread /= static_cast<float>(distances.size());
//
//             if (spread > best_spread)
//             {
//                 best_spread = spread;
//                 best_vp = vp;
//             }
//         }
//
//         p = contained_points[VP_rand_samples[best_vp]];
//         std::ranges::iter_swap(contained_points.begin() + VP_rand_samples[best_vp], contained_points.end() - 1);
//         contained_points.pop_back();
//     }
//
//     void set_median_distance(const std::vector<std::vector<float>>& points, const std::vector<size_t>&
//     contained_points)
//     {
//         if (contained_points.empty()) return;
//
//         std::vector<float> distances;
//         distances.reserve(contained_points.size());
//
//         for (size_t cp : contained_points)
//         {
//             distances.push_back(VPPoint::get_dist(points[p], points[cp]));
//         }
//
//         mu = VPPoint::get_median(distances);
//     }
// };
