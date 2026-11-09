#pragma once

#include <FMM_tSNE/morton/morton.h>
#include <FMM_tSNE/timer/timer.h>
#include <FMM_tSNE/tsne/points/tsne_point_2D.h>
#include <FMM_tSNE/tsne/solvers/SFMM/kernel_SFMM.h>
#include <FMM_tSNE/tsne/solvers/SFMM/node_SFMM.h>
#include <FMM_tSNE/tsne/solvers/solver.h>
#include <algorithm>
#include <boost/sort/sort.hpp>
#include <cstdint>
#include <glm/ext/vector_double2.hpp>
#include <glm/glm.hpp>
#include <glm/gtx/component_wise.hpp>
#include <glm/gtx/string_cast.hpp>
#include <vector>

template <typename T>
class SolverSFMM : public NBodySolver<T>
{
  public:
    std::vector<NodeSFMM> nodes;
    std::vector<unsigned char> depths;  // depth of the nodes parallel to 'nodes'
    std::vector<glm::dvec2> BB_centres; // centres of the nodes parrallel to 'nodes'

    SolverSFMM() = default;

    SolverSFMM(int initMaxChildren, double initTheta)
    {
        this->maxChildren = initMaxChildren;
        this->theta = initTheta;
    }

    void solveNbody(double& total, std::vector<T>& points) override
    {
        NodeSFMM& root = nodes[0];
        traverse_SYM_NN(total, points, root, root, this->theta * this->theta);

        root.field.normalize(1.0 / root.M0);
        applyForces(points, root);
    }

    void updateTree(std::vector<T>& points, glm::dvec2 minPos, glm::dvec2 maxPos) override
    {
        glm::dvec2 negMinPos = -minPos;
        double largestAxis = glm::compMax(maxPos + negMinPos);

#ifdef INDEX_TRACKER
        Morton::createMortonCode(points, negMinPos, largestAxis);
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
    void traverse_SYM_NN(double& total, std::vector<T>& points, NodeSFMM& node_A, NodeSFMM& node_B, double theta_sq)
    {
        // check for node self interaction so it neededn't be checked later
        // self interaction also doesnt require theta check since it will never pass
        if (&node_A == &node_B)
        {
            // if node is leaf then do symmetric PP interactions
            if (node_A.child_count == 0)
            {
                const unsigned end = node_A.first_point_index + node_A.point_count;
                for (unsigned i = node_A.first_point_index; i < end; ++i)
                    for (unsigned j = i + 1; j < end; ++j)
                        TSNE_SFMM_PP(total, points[i], points[j]);
            }
            // if node is not leaf then do each NN interaction that is unique
            else
            {
                const unsigned end = node_A.first_child_index + node_A.child_count;
                for (unsigned i = node_A.first_child_index; i < end; ++i)
                    for (unsigned j = i; j < end; ++j)
                        traverse_SYM_NN(total, points, nodes[i], nodes[j], theta_sq);
            }

            return;
        }

        glm::dvec2 diff = node_A.COM - node_B.COM;
        double length_sum = (node_A.BB_length + node_B.BB_length);

        // efficient theta check
        if (glm::dot(diff, diff) * theta_sq > length_sum * length_sum)
        {
            TSNE_SFMM_NN(total, node_A, node_B);
        }
        // if A is leaf then do PN interaction with B for every point in A
        else if (node_A.child_count == 0)
        {
            for (int i = node_A.first_point_index; i < node_A.first_point_index + node_A.point_count; i++)
            {
                traverse_SYM_PN(total, points, points[i], node_B, theta_sq);
            }
        }
        // if B is leaf then do PN interaction with A for every point in B
        else if (node_B.child_count == 0)
        {
            for (int i = node_B.first_point_index; i < node_B.first_point_index + node_B.point_count; i++)
            {
                traverse_SYM_PN(total, points, points[i], node_A, theta_sq);
            }
        }
        // if B and A are not leaf then make all children interact
        else
        {
            for (int i = node_A.first_child_index; i < node_A.first_child_index + node_A.child_count; i++)
            {
                for (int j = node_B.first_child_index; j < node_B.first_child_index + node_B.child_count; j++)
                {
                    traverse_SYM_NN(total, points, nodes[i], nodes[j], theta_sq);
                }
            }
        }
    }

    void traverse_SYM_PN(double& total, std::vector<T>& points, T& point, NodeSFMM& node, double theta_sq)
    {
        glm::dvec2 diff = point.position - node.COM;
        double length_sum = node.BB_length;

        // efficient theta check
        if (glm::dot(diff, diff) * theta_sq > length_sum * length_sum)
        {
            TSNE_SFMM_PN(total, point, node);
        }
        // if node is leaf then do PP interaction with point for every point in node
        else if (node.child_count == 0)
        {
            for (int i = node.first_point_index; i < node.first_point_index + node.point_count; i++)
            {
                TSNE_SFMM_PP(total, point, points[i]);
            }
        }
        // if B is not leaf then make all children interact with point
        else
        {
            for (int i = node.first_child_index; i < node.first_child_index + node.child_count; i++)
            {
                traverse_SYM_PN(total, points, point, nodes[i], theta_sq);
            }
        }
    }

    void create_tree(std::vector<T>& points, glm::dvec2 minPos, double largestAxis)
    {
        constexpr unsigned max_particles_per_leaf = 32;
        constexpr unsigned max_depth = 16; // 32-bit Morton code = 2 bits per level

        nodes.clear();
        depths.clear();
        BB_centres.clear();
        if (points.empty()) return;

        NodeSFMM root(largestAxis, 0, static_cast<unsigned>(points.size()));

        nodes.push_back(root);
        depths.push_back(0);
        BB_centres.push_back(minPos + 0.5 * largestAxis);

        // nodes grows while we iterate. BFS order keeps each node's children contiguous
        for (size_t n = 0; n < nodes.size(); ++n)
        {
            // copy what we need: push_back below may invalidate references into `nodes`
            const unsigned first = nodes[n].first_point_index;
            const unsigned count = nodes[n].point_count;
            const unsigned depth = depths[n];
            const glm::dvec2 centre = BB_centres[n]; // nodes[n].BB_centre;
            const double length = nodes[n].BB_length;

            // if n is leaf then dont divide further
            if (count <= max_particles_per_leaf || depth >= max_depth)
            {
                nodes[n].first_child_index = 0;
                nodes[n].child_count = 0;
                continue;
            }

            // children of n differ in the 2 bits right after the first 2*depth bits (the node's prefix).
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

                NodeSFMM child(child_length, bounds[q], bounds[q + 1] - bounds[q]);

                nodes.push_back(child);
                depths.push_back(static_cast<unsigned char>(depth + 1));
                BB_centres.push_back(centre + glm::dvec2((q & 1u) ? offset : -offset, (q & 2u) ? offset : -offset));
                ++child_count;
            }
            nodes[n].child_count = child_count;
        }
    }

    void compute_mass_distribution(const std::vector<T>& points)
    {
        // weird loop to avoid size_t underflow
        for (size_t n = nodes.size(); n-- > 0;)
        {
            NodeSFMM& node = nodes[n];

            // leaf node
            if (node.child_count == 0)
            {
                // calculate M0 and COM
                glm::dvec2 sum{0.0};
                for (unsigned i = 0; i < node.point_count; ++i)
                    sum += points[node.first_point_index + i].position;

                node.M0 = static_cast<double>(node.point_count); // unit mass per point
                node.COM = sum / node.M0;

                // calculate M2
                for (unsigned p = node.first_point_index; p < node.first_point_index + node.point_count; ++p)
                {
                    glm::dvec2 relativeCoord = points[p].position - node.COM;

                    node.M2xx += relativeCoord.x * relativeCoord.x;
                    node.M2xy += relativeCoord.x * relativeCoord.y;
                    node.M2yy += relativeCoord.y * relativeCoord.y;
                }
            }
            // non leaf with only 1 child therefore we can prune
            else if (node.child_count == 1)
            {
                // makes node BB visualization correct but costs some performance
                // BB_centres[n] = BB_centres[node.first_child_index];

                node = nodes[node.first_child_index];
            }
            // non leaf, calculate mass distribution from children
            else
            {
                // calculate M0 and COM
                double mass = 0.0;
                glm::dvec2 weighted{0.0};
                for (unsigned c{node.first_child_index}; c < node.first_child_index + node.child_count; ++c)
                {
                    const NodeSFMM& child = nodes[c];

                    mass += child.M0;
                    weighted += child.M0 * child.COM;
                }
                node.M0 = mass;
                node.COM = weighted / mass;

                // calculate M2
                for (unsigned c{node.first_child_index}; c < node.first_child_index + node.child_count; ++c)
                {
                    const NodeSFMM& child = nodes[c];

                    glm::dvec2 relativeCoord = child.COM - node.COM;

                    node.M2xx += child.M0 * relativeCoord.x * relativeCoord.x;
                    node.M2xy += child.M0 * relativeCoord.x * relativeCoord.y;
                    node.M2yy += child.M0 * relativeCoord.y * relativeCoord.y;

                    node.M2xx += child.M2xx;
                    node.M2xy += child.M2xy;
                    node.M2yy += child.M2yy;
                }
            }
        }
    }

    void applyForces(std::vector<T>& points, NodeSFMM& node)
    {
        if (node.child_count != 0)
        {
            for (int c = node.first_child_index; c < node.first_child_index + node.child_count; c++)
            {
                NodeSFMM& child = nodes[c];

                child.field.normalize(1.0 / child.M0);
                child.field += node.field.translate_field_from_to_node(node.COM, child.COM);

                applyForces(points, nodes[c]);
            }
        }
        else
        {
            for (int p = node.first_point_index; p < node.first_point_index + node.point_count; p++)
            {
                T& point = points[p];
                point.derivative += node.field.translate_field_from_to_point(node.COM, point.position);
            }
        }
    }

    void getNodesBufferData(std::vector<VertexPos2Col3>& nodesBufferData, const NodeSFMM&, int, int showLevel)
    {
        for (size_t i{}; i < nodes.size(); ++i)
        {
            const auto& node = nodes[i];
            const unsigned level = depths[i];
            const glm::dvec2 BB_centre = BB_centres[i];

            if (level == showLevel || showLevel == -1)
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

                glm::vec2 lowestCorner = BB_centre - 0.5f * node.BB_length;
                glm::vec2 highestCorner = BB_centre + 0.5f * node.BB_length;

                nodesBufferData.push_back(VertexPos2Col3(glm::vec2(lowestCorner.x, lowestCorner.y), color));
                nodesBufferData.push_back(VertexPos2Col3(glm::vec2(highestCorner.x, lowestCorner.y), color));

                nodesBufferData.push_back(VertexPos2Col3(glm::vec2(lowestCorner.x, lowestCorner.y), color));
                nodesBufferData.push_back(VertexPos2Col3(glm::vec2(lowestCorner.x, highestCorner.y), color));

                nodesBufferData.push_back(VertexPos2Col3(glm::vec2(lowestCorner.x, highestCorner.y), color));
                nodesBufferData.push_back(VertexPos2Col3(glm::vec2(highestCorner.x, highestCorner.y), color));

                nodesBufferData.push_back(VertexPos2Col3(glm::vec2(highestCorner.x, lowestCorner.y), color));
                nodesBufferData.push_back(VertexPos2Col3(glm::vec2(highestCorner.x, highestCorner.y), color));
            }
        }
    }
};
