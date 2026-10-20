#pragma once

#include <FMM_tSNE/timer/timer.h>
#include <FMM_tSNE/tsne/points/tsne_point_2D.h>
#include <FMM_tSNE/tsne/solvers/sFMM/node_sFMM.h>
#include <FMM_tSNE/tsne/solvers/sFMM_fast/kernel_sFMM_fast.h>
#include <FMM_tSNE/tsne/solvers/sFMM_fast/node_sFMM_fast.h>
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
class SolverSFMMFast : public NBodySolver<T>
{
  public:
    std::vector<NodeSFMMFast> nodes;
    std::vector<unsigned char> depths;  // depth of the nodes parallel to 'nodes'
    std::vector<glm::dvec2> BB_centres; // centres of the nodes parrallel to 'nodes'

    SolverSFMMFast() = default;

    SolverSFMMFast(int initMaxChildren, double initTheta)
    {
        this->maxChildren = initMaxChildren;
        this->theta = initTheta;
    }

    void solveNbody(double& total, std::vector<T>& points) override
    {
        traverse_SYM_NN(total, points, nodes[0], nodes[0], this->theta * this->theta);

        double one_over_M0 = 1.0 / nodes[0].M0;
        nodes[0].C2xx *= one_over_M0;
        nodes[0].C2xy *= one_over_M0;
        nodes[0].C2yy *= one_over_M0;
        nodes[0].C3xxx *= one_over_M0;
        nodes[0].C3xxy *= one_over_M0;
        nodes[0].C3xyy *= one_over_M0;
        nodes[0].C3yyy *= one_over_M0;
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
    traverse_SYM_NN(double& total, std::vector<T>& points, NodeSFMMFast& node_A, NodeSFMMFast& node_B, double theta_sq)
    {
        // check for node self interaction so it neededn't be checked later
        // self interaction also doesnt require theta check since it will never pass
        if (&node_A == &node_B)
        {
            // if node is leaf then do symmetric PP interactions
            if (node_A.child_index_amount == 0)
            {
                const unsigned end = node_A.first_particle_index + node_A.particle_index_amount;
                for (unsigned i = node_A.first_particle_index; i < end; ++i)
                    for (unsigned j = i + 1; j < end; ++j)
                        TSNE_sFMMfast_PP(total, points[i], points[j]);
            }
            // if node is not leaf then do each NN interaction that is unique
            else
            {
                const unsigned end = node_A.first_child_index + node_A.child_index_amount;
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
            TSNE_sFMMfast_NN(total, node_A, node_B);
        }
        // if A is leaf then do PN interaction with B for every point in A
        else if (node_A.child_index_amount == 0)
        {
            for (int firstNodePointIndex = node_A.first_particle_index;
                 firstNodePointIndex < node_A.first_particle_index + node_A.particle_index_amount;
                 firstNodePointIndex++)
            {
                traverse_SYM_PN(total, points, points[firstNodePointIndex], node_B, theta_sq);
            }
        }
        // if B is leaf then do PN interaction with A for every point in B
        else if (node_B.child_index_amount == 0)
        {
            for (int secondNodePointIndex = node_B.first_particle_index;
                 secondNodePointIndex < node_B.first_particle_index + node_B.particle_index_amount;
                 secondNodePointIndex++)
            {
                traverse_SYM_PN(total, points, points[secondNodePointIndex], node_A, theta_sq);
            }
        }
        // if B and A are not leaf then make all children interact
        else
        {
            for (int firstNodeChildIndex = node_A.first_child_index;
                 firstNodeChildIndex < node_A.first_child_index + node_A.child_index_amount;
                 firstNodeChildIndex++)
            {
                for (int secondNodeChildIndex = node_B.first_child_index;
                     secondNodeChildIndex < node_B.first_child_index + node_B.child_index_amount;
                     secondNodeChildIndex++)
                {
                    traverse_SYM_NN(total, points, nodes[firstNodeChildIndex], nodes[secondNodeChildIndex], theta_sq);
                }
            }
        }
    }

    void traverse_SYM_PN(double& total, std::vector<T>& points, T& point, NodeSFMMFast& node, double theta_sq)
    {
        glm::dvec2 diff = point.position - node.COM;
        double length_sum = node.BB_length;

        // efficient theta check
        if (glm::dot(diff, diff) * theta_sq > length_sum * length_sum)
        {
            TSNE_sFMMfast_PN(total, point, node);
        }
        // if node is leaf then do PP interaction with point for every point in node
        else if (node.child_index_amount == 0)
        {
            for (int secondNodePointIndex = node.first_particle_index;
                 secondNodePointIndex < node.first_particle_index + node.particle_index_amount;
                 secondNodePointIndex++)
            {
                TSNE_sFMMfast_PP(total, point, points[secondNodePointIndex]);
            }
        }
        // if B is not leaf then make all children interact with point
        else
        {
            for (int secondNodeChildIndex = node.first_child_index;
                 secondNodeChildIndex < node.first_child_index + node.child_index_amount;
                 secondNodeChildIndex++)
            {
                traverse_SYM_PN(total, points, point, nodes[secondNodeChildIndex], theta_sq);
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

        NodeSFMMFast root(largestAxis, 0, static_cast<unsigned>(points.size()));

        nodes.push_back(root);
        depths.push_back(0);
        BB_centres.push_back(minPos + 0.5 * largestAxis);

        // nodes grows while we iterate. BFS order keeps each node's children contiguous
        for (size_t n = 0; n < nodes.size(); ++n)
        {
            // copy what we need: push_back below may invalidate references into `nodes`
            const unsigned first = nodes[n].first_particle_index;
            const unsigned count = nodes[n].particle_index_amount;
            const unsigned depth = depths[n];
            const glm::dvec2 centre = BB_centres[n]; // nodes[n].BB_centre;
            const double length = nodes[n].BB_length;

            // if n is leaf then dont divide further
            if (count <= max_particles_per_leaf || depth >= max_depth)
            {
                nodes[n].first_child_index = 0;
                nodes[n].child_index_amount = 0;
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

                NodeSFMMFast child(child_length, bounds[q], bounds[q + 1] - bounds[q]);

                nodes.push_back(child);
                depths.push_back(static_cast<unsigned char>(depth + 1));
                BB_centres.push_back(centre + glm::dvec2((q & 1u) ? offset : -offset, (q & 2u) ? offset : -offset));
                ++child_count;
            }
            nodes[n].child_index_amount = child_count;
        }
    }

    void compute_mass_distribution(const std::vector<T>& points)
    {
        // weird loop to avoid size_t underflow
        for (size_t n = nodes.size(); n-- > 0;)
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

                    nodes[n].M2xx += relativeCoord.x * relativeCoord.x;
                    nodes[n].M2xy += relativeCoord.x * relativeCoord.y;
                    nodes[n].M2yy += relativeCoord.y * relativeCoord.y;
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

                    // hmmm im not sure about nodes[node.first_child_index + c].M0 CHECK the algorithm
                    // again!!!!!!!!!!!!!!!!!!!!!!!
                    nodes[n].M2xx += nodes[node.first_child_index + c].M0 * relativeCoord.x * relativeCoord.x;
                    nodes[n].M2xy += nodes[node.first_child_index + c].M0 * relativeCoord.x * relativeCoord.y;
                    nodes[n].M2yy += nodes[node.first_child_index + c].M0 * relativeCoord.y * relativeCoord.y;

                    nodes[n].M2xx += nodes[node.first_child_index + c].M2xx;
                    nodes[n].M2xy += nodes[node.first_child_index + c].M2xy;
                    nodes[n].M2yy += nodes[node.first_child_index + c].M2yy;
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
        if (node.child_index_amount != 0)
        {
            for (int child = node.first_child_index; child < node.first_child_index + node.child_index_amount; child++)
            {
                double one_over_M0 = 1.0 / nodes[child].M0;
                nodes[child].C2xx *= one_over_M0;
                nodes[child].C2xy *= one_over_M0;
                nodes[child].C2yy *= one_over_M0;
                nodes[child].C3xxx *= one_over_M0;
                nodes[child].C3xxy *= one_over_M0;
                nodes[child].C3xyy *= one_over_M0;
                nodes[child].C3yyy *= one_over_M0;

                glm::dvec2 oldZ = nodes[child].COM;
                glm::dvec2 newZ = node.COM;
                glm::dvec2 diff = oldZ - newZ;

                double diff_einsum_C2_x = diff.x * node.C2xx + diff.y * node.C2xy;
                double diff_einsum_C2_y = diff.x * node.C2xy + diff.y * node.C2yy;

                double diff_einsum_C3_xx = diff.x * node.C3xxx + diff.y * node.C3xxy;
                double diff_einsum_C3_xy = diff.x * node.C3xxy + diff.y * node.C3xyy;
                double diff_einsum_C3_yy = diff.x * node.C3xyy + diff.y * node.C3yyy;

                double diff_einsumx2_C3_x = diff.x * diff_einsum_C3_xx + diff.y * diff_einsum_C3_xy;
                double diff_einsumx2_C3_y = diff.x * diff_einsum_C3_xy + diff.y * diff_einsum_C3_yy;

                nodes[child].C1x += node.C1x + diff_einsum_C2_x + 0.5 * diff_einsumx2_C3_x;
                nodes[child].C1y += node.C1y + diff_einsum_C2_y + 0.5 * diff_einsumx2_C3_y;
                nodes[child].C2xx += node.C2xx + diff_einsum_C3_xx;
                nodes[child].C2xy += node.C2xy + diff_einsum_C3_xy;
                nodes[child].C2yy += node.C2yy + diff_einsum_C3_yy;
                nodes[child].C3xxx += node.C3xxx;
                nodes[child].C3xxy += node.C3xxy;
                nodes[child].C3xyy += node.C3xyy;
                nodes[child].C3yyy += node.C3yyy;

                applyForces(points, nodes[child]);
            }
        }
        else
        {
            for (int point = node.first_particle_index; point < node.first_particle_index + node.particle_index_amount;
                 point++)
            {
                glm::dvec2 x = points[point].position;
                glm::dvec2 Z0 = node.COM;
                glm::dvec2 diff = points[point].position - node.COM;

                double diff_einsum_C2_x = diff.x * node.C2xx + diff.y * node.C2xy;
                double diff_einsum_C2_y = diff.x * node.C2xy + diff.y * node.C2yy;

                double diff_einsum_C3_xx = diff.x * node.C3xxx + diff.y * node.C3xxy;
                double diff_einsum_C3_xy = diff.x * node.C3xxy + diff.y * node.C3xyy;
                double diff_einsum_C3_yy = diff.x * node.C3xyy + diff.y * node.C3yyy;

                double diff_einsumx2_C3_x = diff.x * diff_einsum_C3_xx + diff.y * diff_einsum_C3_xy;
                double diff_einsumx2_C3_y = diff.x * diff_einsum_C3_xy + diff.y * diff_einsum_C3_yy;

                points[point].derivative.x += node.C1x + diff_einsum_C2_x + 0.5 * diff_einsumx2_C3_x;
                points[point].derivative.y += node.C1y + diff_einsum_C2_y + 0.5 * diff_einsumx2_C3_y;
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

    void
    getNodesBufferData(std::vector<VertexPos2Col3>& nodesBufferData, const NodeSFMMFast& node, int level, int showLevel)
    {
        /*
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
        */
    }
};
