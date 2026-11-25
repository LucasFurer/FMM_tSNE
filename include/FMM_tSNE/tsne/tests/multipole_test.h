#pragma once

#include <FMM_tSNE/data_handling/data_loading/MNIST_digit_data.h>
#include <FMM_tSNE/data_handling/data_processing/data_processing.h>
#include <FMM_tSNE/tsne/points/tsne_point_2D.h>
#include <FMM_tSNE/tsne/solvers/P_matrix/P_matrix.h>
#include <FMM_tSNE/tsne/solvers/SFMM/node_SFMM.h>
#include <FMM_tSNE/tsne/solvers/SFMM/solver_SFMM.h>
#include <FMM_tSNE/tsne/tsne.h>
#include <glm/glm.hpp>
#include <print>
#include <vector>

namespace MultipoleTest
{
inline void test_construction()
{
    constexpr size_t data_size = 1000;
    constexpr float perplexity = 30.0f;
    constexpr unsigned int seed = 327493463u;

    PMatrix p_matrix = PMatrix(DataProcessing::PCA(MNIST_digit_data::load_MNIST_images_fused(data_size), 50),
                               MNIST_digit_data::load_MNIST_labels_fused(data_size),
                               "MNIST_digits",
                               30.0f);
    TSNE tsne(0.75, // min_theta
              0.75, // max_theta
              1.25, // cell_size
              p_matrix,
              seed // seed: 216308u, 592340823u, 4523u, 296343u
    );

    const auto& points = tsne.embeddedPoints;

    NodeSFMM& MP = reinterpret_cast<SolverSFMM<TsnePoint2D>*>(tsne.nBodySolvers["FMM_SYM_MORTON"])->nodes[0];

    NodeSFMM naive_MP = NodeSFMM(0, 0, points.size());

    for (const TsnePoint2D& point : points)
    {
        naive_MP.COM += point.position;
    }
    naive_MP.COM /= static_cast<double>(points.size());
    for (const TsnePoint2D& point : points)
    {
        glm::dvec2 relativeCoord = point.position - naive_MP.COM;

        naive_MP.M2xx += relativeCoord.x * relativeCoord.x;
        naive_MP.M2xy += relativeCoord.x * relativeCoord.y;
        naive_MP.M2yy += relativeCoord.y * relativeCoord.y;
    }

    std::println("COM_naive: {} {}, COM_computed: {} {}", naive_MP.COM.x, naive_MP.COM.y, MP.COM.x, MP.COM.y);
    std::println("COM_diff: {} {}", naive_MP.COM.x - MP.COM.x, naive_MP.COM.y - MP.COM.y);

    std::println("M2_naive: {} {} {}, M2_computed: {} {} {}",
                 naive_MP.M2xx,
                 naive_MP.M2xy,
                 naive_MP.M2yy,
                 MP.M2xx,
                 MP.M2xy,
                 MP.M2yy);
    std::println("M2_diff: {} {} {}", naive_MP.M2xx - MP.M2xx, naive_MP.M2xy - MP.M2xy, naive_MP.M2yy - MP.M2yy);
}
} // namespace MultipoleTest
