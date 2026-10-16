#pragma once

#include <FMM_tSNE/timer/timer.h>
#include <FMM_tSNE/tsne/points/tsne_point_2D.h>
#include <FMM_tSNE/tsne/solvers/sFMM/node_sFMM.h>
#include <FMM_tSNE/tsne/solvers/sFMM_fast/node_sFMM_fast.h>
#include <FMM_tSNE/tsne/solvers/solver.h>
#include <Fastor/Fastor.h>
#include <algorithm>
#include <boost/sort/sort.hpp>
#include <cstdint>
#include <functional>
#include <glm/glm.hpp>
#include <glm/gtx/component_wise.hpp>
#include <glm/gtx/string_cast.hpp>
#include <vector>

template <typename T>
class SolverSFMMFast : public NBodySolver<T>
{
  public:
    std::vector<NodeSFMMFast> nodes; // leaf_nodes

    std::function<void(double&, NodeSFMMFast&, NodeSFMMFast&)> kernelNN;
    std::function<void(double&, T&, NodeSFMMFast&)> kernelPN;
    std::function<void(double&, T&, T&)> kernelPP;

    SolverSFMMFast() = default;

    SolverSFMMFast(std::function<void(double&, NodeSFMMFast&, NodeSFMMFast&)> initKernelNN,
                   std::function<void(double&, T&, NodeSFMMFast&)> initKernelPN,
                   std::function<void(double&, T&, T&)> initKernelPP,
                   int initMaxChildren,
                   double initTheta)
    {
        kernelNN = initKernelNN;
        kernelPN = initKernelPN;
        kernelPP = initKernelPP;
        this->maxChildren = initMaxChildren;
        this->theta = initTheta;
    }

    void solveNbody(double& total, std::vector<T>& points) override
    {
        traverse_SYM_NN(total, points, nodes[0], nodes[0], this->theta);

        double one_over_M0 = 1.0 / nodes[0].M0;
        nodes[0].C2 *= one_over_M0;
        nodes[0].C3 *= one_over_M0;
        applyForces(points, nodes[0]);
    }

    void updateTree(std::vector<T>& points, glm::dvec2 minPos, glm::dvec2 maxPos) override
    {
        glm::dvec2 negMinPos = -minPos;
        double largestAxis = glm::compMax(maxPos + negMinPos);

#ifdef INDEX_TRACKER
        createMortonCode(points, negMinPos, largestAxis);
        boost::sort::spreadsort::integer_sort(
            points.begin(), points.end(), TsnePoint2DRightshift(), TsnePoint2DLessthan());
#endif

        create_tree(points, minPos, largestAxis);
        compute_mass_distribution(points);
    }

    std::vector<VertexPos2Col3> getNodesBufferData(int nodeLevelToShow) override
    {
        std::vector<VertexPos2Col3> result;

        getNodesBufferData(result, nodes[0], 0, nodeLevelToShow);

        return result;
    }

  private:
    void
    traverse_SYM_NN(double& total, std::vector<T>& points, NodeSFMMFast& node_A, NodeSFMMFast& node_B, double theta)
    {
        glm::dvec2 diff = node_A.COM - node_B.COM;
        // double dist = glm::length(diff);
        double dist = glm::dot(diff, diff);

        double threshold = (node_A.BB_length + node_B.BB_length) / theta;
        if (dist > threshold * threshold) // sym node node
        // if ((node_A.BB_length + node_B.BB_length) / dist < theta) // sym node node
        {
            kernelNN(total, node_A, node_B);
        }
        else if (node_A.child_index_amount == 0) // traverse for each point in first
        {

            if (&node_A == &node_B)
            {
                for (int firstNodePointIndex = node_A.first_particle_index;
                     firstNodePointIndex < node_A.first_particle_index + node_A.particle_index_amount;
                     firstNodePointIndex++)
                {
                    for (int secondNodePointIndex = firstNodePointIndex;
                         secondNodePointIndex < node_B.first_particle_index + node_B.particle_index_amount;
                         secondNodePointIndex++)
                    {
                        if (points[firstNodePointIndex].ID != points[secondNodePointIndex].ID)
                        {
                            // can we exclude the PP self interactions somehow in the for loops?
                            kernelPP(total, points[firstNodePointIndex], points[secondNodePointIndex]);
                        }
                    }
                }
            }
            else
            {
                for (int firstNodePointIndex = node_A.first_particle_index;
                     firstNodePointIndex < node_A.first_particle_index + node_A.particle_index_amount;
                     firstNodePointIndex++)
                {
                    traverse_SYM_PN(total, points, points[firstNodePointIndex], node_B, theta);
                }
            }
        }
        else if (node_B.child_index_amount == 0) // traverse for each point in second
        {

            if (&node_A == &node_B)
            {
                for (int secondNodePointIndex = node_B.first_particle_index;
                     secondNodePointIndex < node_B.first_particle_index + node_B.particle_index_amount;
                     secondNodePointIndex++)
                {
                    for (int firstNodePointIndex = secondNodePointIndex;
                         firstNodePointIndex < node_A.first_particle_index + node_A.particle_index_amount;
                         firstNodePointIndex++)
                    {
                        if (points[firstNodePointIndex].ID != points[secondNodePointIndex].ID)
                        {
                            // can we exclude the PP self interactions somehow in the for loops?
                            kernelPP(total, points[firstNodePointIndex], points[secondNodePointIndex]);
                        }
                    }
                }
            }
            else
            {
                for (int secondNodePointIndex = node_B.first_particle_index;
                     secondNodePointIndex < node_B.first_particle_index + node_B.particle_index_amount;
                     secondNodePointIndex++)
                {
                    traverse_SYM_PN(total, points, points[secondNodePointIndex], node_A, theta);
                }
            }
        }
        else // traverse for each node in first and second
        {

            if (&node_A == &node_B) // if nodes from interaction pair are the same then dont add duplicates
            {
                for (int firstNodeChildIndex = node_A.first_child_index;
                     firstNodeChildIndex < node_A.first_child_index + node_A.child_index_amount;
                     firstNodeChildIndex++)
                {
                    for (int secondNodeChildIndex = firstNodeChildIndex;
                         secondNodeChildIndex < node_A.first_child_index + node_A.child_index_amount;
                         secondNodeChildIndex++)
                    {
                        traverse_SYM_NN(total, points, nodes[firstNodeChildIndex], nodes[secondNodeChildIndex], theta);
                    }
                }
            }
            else // if nodes from interaction pair are the NOT same then add all combinations
            {
                for (int firstNodeChildIndex = node_A.first_child_index;
                     firstNodeChildIndex < node_A.first_child_index + node_A.child_index_amount;
                     firstNodeChildIndex++)
                {
                    for (int secondNodeChildIndex = node_B.first_child_index;
                         secondNodeChildIndex < node_B.first_child_index + node_B.child_index_amount;
                         secondNodeChildIndex++)
                    {
                        traverse_SYM_NN(total, points, nodes[firstNodeChildIndex], nodes[secondNodeChildIndex], theta);
                    }
                }
            }
        }
    }

    void traverse_SYM_PN(double& total, std::vector<T>& points, T& firstPoint, NodeSFMMFast& secondNode, double theta)
    {
        glm::dvec2 diff = firstPoint.position - secondNode.COM;
        // double dist = glm::length(diff);
        double dist = glm::dot(diff, diff);

        // double threshold = (secondNode.BB_length) / (theta * 0.35);
        double threshold = secondNode.BB_length / theta;
        if (dist > threshold * threshold) // sym point node
        // if (secondNode.BB_length / dist < theta)
        {
            kernelPN(total, firstPoint, secondNode);
        }
        else if (secondNode.child_index_amount == 0) // sym point point
        {

            for (int secondNodePointIndex = secondNode.first_particle_index;
                 secondNodePointIndex < secondNode.first_particle_index + secondNode.particle_index_amount;
                 secondNodePointIndex++)
            {
                if (points[secondNodePointIndex].ID != firstPoint.ID)
                {
                    kernelPP(total, firstPoint, points[secondNodePointIndex]);
                }
            }
        }
        else // traverse for each node in second
        {

            for (int secondNodeChildIndex = secondNode.first_child_index;
                 secondNodeChildIndex < secondNode.first_child_index + secondNode.child_index_amount;
                 secondNodeChildIndex++)
            {
                traverse_SYM_PN(total, points, firstPoint, nodes[secondNodeChildIndex], theta);
            }
        }
    }

    void divide_by_mass()
    {
        for (int i = 0; i < nodes.size(); i++)
        {
            if (nodes[i].M0 != 0.0)
            {
                // nodes[i].C1 /= nodes[i].M0;
                nodes[i].C2 /= nodes[i].M0;
                nodes[i].C3 /= nodes[i].M0;
            }
        }
    }

    void create_tree(std::vector<T>& points, glm::dvec2 minPos, double largestAxis)
    {
        constexpr unsigned max_particles_per_leaf = 32;
        constexpr unsigned max_depth = 16; // 32-bit Morton code = 2 bits per level

        nodes.clear();
        if (points.empty()) return;

        // depth of every node, parallel to `nodes` (NodeSFMMFast doesn't store it)
        std::vector<unsigned char> depths;

        NodeSFMMFast root(minPos + 0.5 * largestAxis, largestAxis, 0, static_cast<unsigned>(points.size()));

        nodes.push_back(root);
        depths.push_back(0);

        // nodes grows while we iterate; BFS order keeps each node's children contiguous
        for (size_t n = 0; n < nodes.size(); ++n)
        {
            // copy what we need: push_back below may invalidate references into `nodes`
            const unsigned first = nodes[n].first_particle_index;
            const unsigned count = nodes[n].particle_index_amount;
            const unsigned depth = depths[n];
            const glm::dvec2 centre = nodes[n].BB_centre;
            const double length = nodes[n].BB_length;

            if (count <= max_particles_per_leaf || depth >= max_depth) continue; // leaf

            // The children of this node differ in the 2 bits right after the first
            // 2*depth bits (the node's prefix).
            const unsigned shift = 32u - 2u * (depth + 1u);

            // uint64_t avoids UB for shift + 2 == 32 at the root
            const std::uint64_t prefix = static_cast<std::uint64_t>(points[first].morton_code) >> (shift + 2u);

            // bounds[q] .. bounds[q+1] is the particle range of quadrant q
            unsigned bounds[5];
            bounds[0] = first;
            bounds[4] = first + count;

            auto end_it = points.begin() + bounds[4];
            for (unsigned q = 1; q < 4; ++q)
            {
                const std::uint64_t target = (prefix << 2) | q;
                // first point whose (prefix + quadrant bits) >= target.
                // Search starts at the previous boundary, so the range shrinks each time.
                auto it = std::partition_point(
                    points.begin() + bounds[q - 1],
                    end_it,
                    [&](const T& p) { return (static_cast<std::uint64_t>(p.morton_code) >> shift) < target; });
                bounds[q] = static_cast<unsigned>(it - points.begin());
            }

            // create the non-empty children
            const double child_length = 0.5 * length;
            const double offset = 0.25 * length;

            nodes[n].first_child_index = static_cast<unsigned>(nodes.size());
            unsigned char child_count = 0;

            for (unsigned q = 0; q < 4; ++q)
            {
                if (bounds[q] == bounds[q + 1]) continue;

                NodeSFMMFast child(centre + glm::dvec2((q & 1u) ? offset : -offset, (q & 2u) ? offset : -offset),
                                   child_length,
                                   bounds[q],
                                   bounds[q + 1] - bounds[q]);

                nodes.push_back(child);
                depths.push_back(static_cast<unsigned char>(depth + 1));
                ++child_count;
            }
            nodes[n].child_index_amount = child_count;
        }
    }

    void compute_mass_distribution(const std::vector<T>& points)
    {
        for (size_t n = nodes.size(); n-- > 0;) // hmmm weird loop why no --n at the end
        {
            NodeSFMMFast& node = nodes[n];

            if (node.child_index_amount == 0)
            {
                // calculate M0 and COM
                glm::dvec2 sum{0.0};
                for (unsigned i = 0; i < node.particle_index_amount; ++i)
                    sum += points[node.first_particle_index + i].position;

                node.M0 = static_cast<double>(node.particle_index_amount); // unit mass per point
                node.COM = sum / node.M0;

                // calculate M2
                for (unsigned i = 0; i < node.particle_index_amount; ++i)
                {
                    glm::dvec2 relativeCoord = points[node.first_particle_index + i].position - nodes[n].COM;

                    Fastor::Tensor<double, 2, 2> outer_product;
                    outer_product(0, 0) = relativeCoord.x * relativeCoord.x;
                    outer_product(0, 1) = relativeCoord.x * relativeCoord.y;
                    outer_product(1, 0) = relativeCoord.y * relativeCoord.x;
                    outer_product(1, 1) = relativeCoord.y * relativeCoord.y;
                    nodes[n].M2 += outer_product;
                }
            }
            else if (node.child_index_amount == 1)
            {
                node = nodes[node.first_child_index];
            }
            else
            {
                // calculate M0 and COM
                double mass = 0.0;
                glm::dvec2 weighted{0.0};
                for (unsigned c = 0; c < node.child_index_amount; ++c)
                {
                    const NodeSFMMFast& child = nodes[node.first_child_index + c];
                    mass += child.M0;
                    weighted += child.M0 * child.COM;
                }
                node.M0 = mass;
                node.COM = weighted / mass;

                // calculate M2
                for (unsigned c = 0; c < node.child_index_amount; ++c)
                {
                    glm::dvec2 relativeCoord = nodes[node.first_child_index + c].COM - nodes[n].COM;

                    Fastor::Tensor<double, 2, 2> outer_product;
                    outer_product(0, 0) = relativeCoord.x * relativeCoord.x;
                    outer_product(0, 1) = relativeCoord.x * relativeCoord.y;
                    outer_product(1, 0) = relativeCoord.y * relativeCoord.x;
                    outer_product(1, 1) = relativeCoord.y * relativeCoord.y;
                    nodes[n].M2 += outer_product;

                    nodes[n].M2 += nodes[c].M2;
                }
            }
        }
    }

    inline uint32_t compact16(uint32_t x)
    {
        x &= 0x55555555u;
        x = (x | (x >> 1)) & 0x33333333u;
        x = (x | (x >> 2)) & 0x0F0F0F0Fu;
        x = (x | (x >> 4)) & 0x00FF00FFu;
        x = (x | (x >> 8)) & 0x0000FFFFu;
        return x;
    }

    void applyForces(std::vector<T>& points, NodeSFMMFast& node)
    {
        // if a nodes first child index = the root (0) then you dont have any children
        if (node.child_index_amount != 0)
        {
            for (int nodeIndex = node.first_child_index; nodeIndex < node.first_child_index + node.child_index_amount;
                 nodeIndex++)
            {
                double one_over_M0 = 1.0 / nodes[nodeIndex].M0;
                nodes[nodeIndex].C2 *= one_over_M0;
                nodes[nodeIndex].C3 *= one_over_M0;

                glm::dvec2 oldZ = nodes[nodeIndex].COM;
                glm::dvec2 newZ = node.COM;
                Fastor::Tensor<double, 2> diff1 = {oldZ.x - newZ.x, oldZ.y - newZ.y};
                // Fastor::Tensor<float, 2, 2> diff2 = Fastor::outer(diff1, diff1);
                // Fastor::Tensor<float, 2, 2, 2> diff3 = Fastor::outer(diff2, diff1);

                Fastor::Tensor<double, 2> newC1 =
                    node.C1 + Fastor::einsum<Fastor::Index<0>, Fastor::Index<0, 1>>(diff1, node.C2) +
                    0.5 * Fastor::einsum<Fastor::Index<0>, Fastor::Index<0, 1>>(
                              diff1, Fastor::einsum<Fastor::Index<0>, Fastor::Index<0, 1, 2>>(diff1, node.C3));

                Fastor::Tensor<double, 2, 2> newC2 =
                    node.C2 + Fastor::einsum<Fastor::Index<0>, Fastor::Index<0, 1, 2>>(diff1, node.C3);

                Fastor::Tensor<double, 2, 2, 2> newC3 = node.C3;

                nodes[nodeIndex].C1 += newC1;
                nodes[nodeIndex].C2 += newC2;
                nodes[nodeIndex].C3 += newC3;

                applyForces(points, nodes[nodeIndex]);
            }
        }
        else
        {
            for (int pointIndex = node.first_particle_index;
                 pointIndex < node.first_particle_index + node.particle_index_amount;
                 pointIndex++)
            {
                glm::dvec2 x = points[pointIndex].position;
                glm::dvec2 Z0 = node.COM;
                Fastor::Tensor<double, 2> diff1 = {x.x - Z0.x, x.y - Z0.y};
                // Fastor::Tensor<float, 2, 2> diff2 = Fastor::outer(diff1, diff1);
                // Fastor::Tensor<float, 2, 2, 2> diff3 = Fastor::outer(diff2, diff1);

                Fastor::Tensor<double, 2> newC1 =
                    node.C1 + Fastor::einsum<Fastor::Index<0>, Fastor::Index<0, 1>>(diff1, node.C2) +
                    0.5 * Fastor::einsum<Fastor::Index<0>, Fastor::Index<0, 1>>(
                              diff1, Fastor::einsum<Fastor::Index<0>, Fastor::Index<0, 1, 2>>(diff1, node.C3));

                points[pointIndex].derivative += glm::dvec2(newC1(0), newC1(1));
            }
        }
    }

    void createMortonCode(std::vector<TsnePoint2D>& points, glm::dvec2 negMinPos, double largestAxis)
    {
        glm::dvec2 position{0.0};
        for (int i = 0; i < points.size(); i++)
        {
            position = points[i].position;

            position = (position + negMinPos) / (largestAxis); // rescale such that range is between [0-1]

            position = glm::min(
                glm::max(position * 65536.0, glm::dvec2(0.0)),
                glm::dvec2(
                    65535.0)); // rescale such that range is between [0-65535] which is the max number for 16 bits
#ifdef INDEX_TRACKER
            points[i].morton_code = (spread16((uint32_t) position.y) << 1) | spread16((uint32_t) position.x);
#endif
        }
    }

    inline uint32_t spread16(uint32_t x)
    {
        x = (x | (x << 8)) & 0x00FF00FFu;
        x = (x | (x << 4)) & 0x0F0F0F0Fu;
        x = (x | (x << 2)) & 0x33333333u;
        x = (x | (x << 1)) & 0x55555555u;
        return x;
    }

    void getNodesBufferData(std::vector<VertexPos2Col3>& nodesBufferData, NodeSFMMFast node, int level, int showLevel)
    {
        if ((level == showLevel || showLevel == -1) && node.particle_index_amount != 0)
        {
            const int colorsSize = 7;
            std::array<glm::vec3, colorsSize> colors{glm::vec3(1.0f, 1.0f, 1.0f),
                                                     glm::vec3(1.0f, 0.0f, 0.0f),
                                                     glm::vec3(0.0f, 1.0f, 0.0f),
                                                     glm::vec3(0.0f, 0.0f, 1.0f),
                                                     glm::vec3(1.0f, 1.0f, 0.0f),
                                                     glm::vec3(0.0f, 1.0f, 1.0f),
                                                     glm::vec3(1.0f, 0.0f, 1.0f)};

            glm::vec3 color = colors[std::min(showLevel + 1, colorsSize - 1)];

            glm::vec2 lowestCorner = node.BB_centre - 0.5f * node.BB_length;
            glm::vec2 highestCorner = node.BB_centre + 0.5f * node.BB_length;

            nodesBufferData.push_back(VertexPos2Col3(glm::vec2(lowestCorner.x, lowestCorner.y), color));
            nodesBufferData.push_back(VertexPos2Col3(glm::vec2(highestCorner.x, lowestCorner.y), color));

            nodesBufferData.push_back(VertexPos2Col3(glm::vec2(lowestCorner.x, lowestCorner.y), color));
            nodesBufferData.push_back(VertexPos2Col3(glm::vec2(lowestCorner.x, highestCorner.y), color));

            nodesBufferData.push_back(VertexPos2Col3(glm::vec2(lowestCorner.x, highestCorner.y), color));
            nodesBufferData.push_back(VertexPos2Col3(glm::vec2(highestCorner.x, highestCorner.y), color));

            nodesBufferData.push_back(VertexPos2Col3(glm::vec2(highestCorner.x, lowestCorner.y), color));
            nodesBufferData.push_back(VertexPos2Col3(glm::vec2(highestCorner.x, highestCorner.y), color));
        }

        if (node.child_index_amount != 0u)
        {
            for (int i = node.first_child_index; i < node.first_child_index + node.child_index_amount; i++)
            {
                getNodesBufferData(nodesBufferData, nodes[i], level + 1, showLevel);
            }
        }
    }
};

// TSNE kernals
// -------------------------------------------------------------
inline void TSNE_FMMfast_SYM_MORTON_NN_Kernel(double& total, NodeSFMMFast& sinkNode, NodeSFMMFast& sourceNode)
{
    glm::dvec2 R = sinkNode.COM - sourceNode.COM;
    double sq_r = R.x * R.x + R.y * R.y;
    double rS = 1.0 + sq_r;

    double inv_rS = 1.0 / rS;
    double inv_rS2 = inv_rS * inv_rS;
    double inv_rS3 = inv_rS2 * inv_rS;
    double inv_rS4 = inv_rS3 * inv_rS;

    double D1 = inv_rS2;
    double D2 = -4.0 * inv_rS3;
    double D3 = 24.0 * inv_rS4;

    double MA0 = sinkNode.M0;
    double MB0 = sourceNode.M0;
    Fastor::Tensor<double, 2, 2> MB2 = sourceNode.M2;
    Fastor::Tensor<double, 2, 2> MB2Tilde = (1.0 / MB0) * MB2;

    // calculate the C^m
    double MB2TildeSum1 = MB2Tilde(0, 0) + MB2Tilde(1, 1);
    double MB2TildeSum2 = (R.x * R.x * MB2Tilde(0, 0)) + (R.x * R.y * MB2Tilde(0, 1)) + (R.y * R.x * MB2Tilde(1, 0)) +
                          (R.y * R.y * MB2Tilde(1, 1));
    double MB2TildeSum3i0 = R.x * MB2Tilde(0, 0) + R.y * MB2Tilde(0, 1);
    double MB2TildeSum3i1 = R.x * MB2Tilde(1, 0) + R.y * MB2Tilde(1, 1);

    Fastor::Tensor<double, 2> C1 = Fastor::Tensor<double, 2>{
        MB0 * (R.x * (D1 + 0.5 * (MB2TildeSum1) *D2 + 0.5 * (MB2TildeSum2) *D3) + (MB2TildeSum3i0) *D2),
        MB0 * (R.y * (D1 + 0.5 * (MB2TildeSum1) *D2 + 0.5 * (MB2TildeSum2) *D3) + (MB2TildeSum3i1) *D2)};

    Fastor::Tensor<double, 2, 2> C2 =
        MA0 * Fastor::Tensor<double, 2, 2>{{MB0 * (D1 + R.x * R.x * D2), MB0 * (R.x * R.y * D2)},
                                           {MB0 * (R.y * R.x * D2), MB0 * (D1 + R.y * R.y * D2)}};

    Fastor::Tensor<double, 2, 2, 2> C3 =
        MA0 * Fastor::Tensor<double, 2, 2, 2>{
                  {{
                       MB0 * ((R.x + R.x + R.x) * D2 + R.x * R.x * R.x * D3), // i = 0, j = 0, k = 0
                       MB0 * ((R.y) * D2 + R.x * R.x * R.y * D3)              // i = 0, j = 0, k = 1
                   },
                   {
                       MB0 * ((R.y) * D2 + R.x * R.y * R.x * D3), // i = 0, j = 1, k = 0
                       MB0 * ((R.x) * D2 + R.x * R.y * R.y * D3)  // i = 0, j = 1, k = 1
                   }},
                  {{
                       MB0 * ((R.y) * D2 + R.y * R.x * R.x * D3), // i = 1, j = 0, k = 0
                       MB0 * ((R.x) * D2 + R.y * R.x * R.y * D3)  // i = 1, j = 0, k = 1
                   },
                   {
                       MB0 * ((R.x) * D2 + R.y * R.y * R.x * D3),            // i = 1, j = 1, k = 0
                       MB0 * ((R.y + R.y + R.y) * D2 + R.y * R.y * R.y * D3) // i = 1, j = 1, k = 1
                   }}};

    sinkNode.C1 += C1;
    sinkNode.C2 += C2;
    sinkNode.C3 += C3;

    // second interaction ---------------------------------------------------------------------------------------

    R = -R;

    MA0 = sourceNode.M0;
    MB0 = sinkNode.M0;
    MB2 = sinkNode.M2;
    MB2Tilde = (1.0 / MB0) * MB2;

    // calculate the C^m
    MB2TildeSum1 = MB2Tilde(0, 0) + MB2Tilde(1, 1);
    MB2TildeSum2 = (R.x * R.x * MB2Tilde(0, 0)) + (R.x * R.y * MB2Tilde(0, 1)) + (R.y * R.x * MB2Tilde(1, 0)) +
                   (R.y * R.y * MB2Tilde(1, 1));
    MB2TildeSum3i0 = R.x * MB2Tilde(0, 0) + R.y * MB2Tilde(0, 1);
    MB2TildeSum3i1 = R.x * MB2Tilde(1, 0) + R.y * MB2Tilde(1, 1);

    C1 = Fastor::Tensor<double, 2>{
        MB0 * (R.x * (D1 + 0.5 * (MB2TildeSum1) *D2 + 0.5 * (MB2TildeSum2) *D3) + (MB2TildeSum3i0) *D2),
        MB0 * (R.y * (D1 + 0.5 * (MB2TildeSum1) *D2 + 0.5 * (MB2TildeSum2) *D3) + (MB2TildeSum3i1) *D2)};

    sourceNode.C1 += C1;
    sourceNode.C2 += C2;
    sourceNode.C3 += -C3;

    // add totals for both interactions
    total += 2.0 * (sinkNode.M0 * sourceNode.M0) / rS;
}

inline void TSNE_FMMfast_SYM_MORTON_PN_Kernel(double& total, TsnePoint2D& sinkPoint, NodeSFMMFast& sourceNode)
{
    glm::dvec2 R = sinkPoint.position - sourceNode.COM;
    double sq_r = R.x * R.x + R.y * R.y;
    double rS = 1.0 + sq_r;

    double inv_rS = 1.0 / rS;
    double inv_rS2 = inv_rS * inv_rS;
    double inv_rS3 = inv_rS2 * inv_rS;
    double inv_rS4 = inv_rS3 * inv_rS;

    double D1 = inv_rS2;
    double D2 = -4.0 * inv_rS3;
    double D3 = 24.0 * inv_rS4;

    double MA0 = 1.0;
    double MB0 = sourceNode.M0;
    Fastor::Tensor<double, 2, 2> MB2 = sourceNode.M2;
    Fastor::Tensor<double, 2, 2> MB2Tilde = (1.0 / MB0) * MB2;

    double MB2TildeSum1 = MB2Tilde(0, 0) + MB2Tilde(1, 1);
    double MB2TildeSum2 = (R.x * R.x * MB2Tilde(0, 0)) + (R.x * R.y * MB2Tilde(0, 1)) + (R.y * R.x * MB2Tilde(1, 0)) +
                          (R.y * R.y * MB2Tilde(1, 1));
    double MB2TildeSum3i0 = R.x * MB2Tilde(0, 0) + R.y * MB2Tilde(0, 1);
    double MB2TildeSum3i1 = R.x * MB2Tilde(1, 0) + R.y * MB2Tilde(1, 1);

    Fastor::Tensor<double, 2> C1 = Fastor::Tensor<double, 2>{
        MB0 * (R.x * (D1 + 0.5 * (MB2TildeSum1) *D2 + 0.5 * (MB2TildeSum2) *D3) + (MB2TildeSum3i0) *D2),
        MB0 * (R.y * (D1 + 0.5 * (MB2TildeSum1) *D2 + 0.5 * (MB2TildeSum2) *D3) + (MB2TildeSum3i1) *D2)};

    sinkPoint.derivative += glm::dvec2(C1(0), C1(1));

    // second interaction ---------------------------------------------------------------------------------------

    R = -R;

    MA0 = sourceNode.M0;
    MB0 = 1.0;

    C1 = Fastor::Tensor<double, 2>{(R.x * D1), (R.y * D1)};

    Fastor::Tensor<double, 2, 2> C2 = MA0 * Fastor::Tensor<double, 2, 2>{{(D1 + R.x * R.x * D2), (R.x * R.y * D2)},
                                                                         {(R.y * R.x * D2), (D1 + R.y * R.y * D2)}};

    Fastor::Tensor<double, 2, 2, 2> C3 =
        MA0 *
        Fastor::Tensor<double, 2, 2, 2>{{{
                                             ((R.x + R.x + R.x) * D2 + R.x * R.x * R.x * D3), // i = 0, j = 0, k = 0
                                             ((R.y) * D2 + R.x * R.x * R.y * D3)              // i = 0, j = 0, k = 1
                                         },
                                         {
                                             ((R.y) * D2 + R.x * R.y * R.x * D3), // i = 0, j = 1, k = 0
                                             ((R.x) * D2 + R.x * R.y * R.y * D3)  // i = 0, j = 1, k = 1
                                         }},
                                        {{
                                             ((R.y) * D2 + R.y * R.x * R.x * D3), // i = 1, j = 0, k = 0
                                             ((R.x) * D2 + R.y * R.x * R.y * D3)  // i = 1, j = 0, k = 1
                                         },
                                         {
                                             ((R.x) * D2 + R.y * R.y * R.x * D3),            // i = 1, j = 1, k = 0
                                             ((R.y + R.y + R.y) * D2 + R.y * R.y * R.y * D3) // i = 1, j = 1, k = 1
                                         }}};

    sourceNode.C1 += C1;
    sourceNode.C2 += C2;
    sourceNode.C3 += C3;

    total += 2.0 * sourceNode.M0 / rS;
}

inline void TSNE_FMMfast_SYM_MORTON_PP_Kernel(double& total, TsnePoint2D& sinkPoint, TsnePoint2D& sourcePoint)
{
    glm::dvec2 R = sinkPoint.position - sourcePoint.position;
    double sq_dist = R.x * R.x + R.y * R.y;

    double forceDecay = 1.0 / (1.0 + sq_dist);

    sinkPoint.derivative += forceDecay * forceDecay * R;

    // second interaction

    R = -R;

    sourcePoint.derivative += forceDecay * forceDecay * R;

    // add totals for both interactions

    total += 2.0 * forceDecay;
}
