#pragma once

#include <FMM_tSNE/tsne/points/tsne_point_2D.h>
#include <FMM_tSNE/tsne/solvers/sFMM/node_sFMM.h>
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

// #include "../../common.h"
// #include "../../particles/Particle2D.h"
// #include "../../particles/embeddedPoint.h"
// #include "../../particles/morton_point.h"
// #include "../../particles/tsnePoint2D.h"
// #include "../../trees/cpu/nodeFMM_MORTON_2D.h"
// #include "../../trees/cpu/quadtreeFMM.h"
// #include "nBodySolver.h"

template <typename T>
class NBodySolverFMM_SYM_MORTON : public NBodySolver<T>
{
  public:
    unsigned int treeDepth;
    std::vector<NodeFMM_MORTON_2D> nodes; // leaf_nodes
    std::vector<unsigned int> levelIndex;
    std::vector<unsigned int> levelSize;
    std::vector<unsigned int> levelGridWidth;

    std::function<void(double&, NodeFMM_MORTON_2D&, NodeFMM_MORTON_2D&)> kernelNN;
    std::function<void(double&, T&, NodeFMM_MORTON_2D&)> kernelPN;
    // std::function<void(double&, NodeFMM_MORTON_2D&, T&)> kernelNP;
    std::function<void(double&, T&, T&)> kernelPP;

    // std::vector<std::pair<NodeFMM_MORTON_2D, NodeFMM_MORTON_2D>> interaction_NN_stack;
    // std::vector<std::pair<T, NodeFMM_MORTON_2D>> interaction_PN_stack;

    NBodySolverFMM_SYM_MORTON() {}

    NBodySolverFMM_SYM_MORTON(std::function<void(double&, NodeFMM_MORTON_2D&, NodeFMM_MORTON_2D&)> initKernelNN,
                              std::function<void(double&, T&, NodeFMM_MORTON_2D&)> initKernelPN,
                              // std::function<void(double&, NodeFMM_MORTON_2D&, T&)> initKernelNP,
                              std::function<void(double&, T&, T&)> initKernelPP,
                              int initMaxChildren,
                              unsigned int initTreeDepth,
                              double initTheta)
    {
        kernelNN = initKernelNN;
        kernelPN = initKernelPN;
        // kernelNP = initKernelNP;
        kernelPP = initKernelPP;
        this->maxChildren = initMaxChildren;
        initNodesSize(initTreeDepth);
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
        NodeFMM_MORTON_2D emptyNode;
        std::fill(nodes.begin(), nodes.end(), emptyNode);

        glm::dvec2 negMinPos = -minPos;
        double largestAxis = glm::compMax(maxPos + negMinPos);

#ifdef INDEX_TRACKER
        createMortonCode(points, negMinPos, largestAxis);
        boost::sort::spreadsort::integer_sort(
            points.begin(), points.end(), TsnePoint2DRightshift(), TsnePoint2DLessthan());
#endif

        createLeafNodes(points, minPos, maxPos);

        bottomUpNodeConstruction(minPos, maxPos);
    }

    std::vector<VertexPos2Col3> getNodesBufferData(int nodeLevelToShow) override
    {
        std::vector<VertexPos2Col3> result;

        getNodesBufferData(result, nodes[0], 0, nodeLevelToShow);

        return result;
    }

    static unsigned int getDepth(int max_children, int N)
    {
        double circle_area = std::numbers::pi / 4.0;
        double depth_real = std::log((1.0 / circle_area) * (N / max_children)) / std::log(4.0);
        unsigned int depth = static_cast<unsigned int>(std::ceil(depth_real));

        return depth;
    }

    void initNodesSize(unsigned int initTreeDepth)
    {
        treeDepth = initTreeDepth;
        levelIndex.resize(treeDepth + 1);
        levelSize.resize(treeDepth + 1);
        levelGridWidth.resize(treeDepth + 1);
        levelIndex[0] = 0;
        int nodesSize = 0;
        for (int i = 0; i <= treeDepth; i++) // treeDepth = 0 means just the root
        {
            int currentLevelSize = 1;
            for (int j = 0; j < i; j++)
            {
                currentLevelSize *= 4;
            }
            levelSize[i] = currentLevelSize;
            nodesSize += currentLevelSize;

            int currentDepthStart = 0;
            for (int j = 0; j <= i; j++)
            {
                currentDepthStart += levelSize[j];
            }
            if (i + 1 < treeDepth + 1) levelIndex[i + 1] = currentDepthStart;
        }

        nodes.resize(nodesSize);

        for (int i = 0; i <= treeDepth; i++)
            levelGridWidth[i] = std::pow(2, i);
        // leafGridSize = std::pow(2, treeDepth);
    }

  private:
    void traverse_SYM_NN(
        double& total, std::vector<T>& points, NodeFMM_MORTON_2D& node_A, NodeFMM_MORTON_2D& node_B, double theta)
    {
        glm::dvec2 diff = node_A.centreOfMass - node_B.centreOfMass;
        // double dist = glm::length(diff);
        double dist = glm::dot(diff, diff);

        double threshold = (node_A.BBlength + node_B.BBlength) / theta;
        if (dist > threshold * threshold) // sym node node
        // if ((node_A.BBlength + node_B.BBlength) / dist < theta) // sym node node
        {

            if (node_A.particleIndexAmount != 0 && node_B.particleIndexAmount != 0)
            {

                kernelNN(total, node_A, node_B);
            }
        }
        else if (node_A.firstChildIndex == 0) // traverse for each point in first
        {

            if (&node_A == &node_B)
            {
                for (int firstNodePointIndex = node_A.firstParticleIndex;
                     firstNodePointIndex < node_A.firstParticleIndex + node_A.particleIndexAmount;
                     firstNodePointIndex++)
                {
                    if (node_B.particleIndexAmount != 0)
                    {
                        for (int secondNodePointIndex = firstNodePointIndex;
                             secondNodePointIndex < node_B.firstParticleIndex + node_B.particleIndexAmount;
                             secondNodePointIndex++)
                        {
                            if (points[firstNodePointIndex].ID != points[secondNodePointIndex].ID)
                            {

                                kernelPP(total,
                                         points[firstNodePointIndex],
                                         points[secondNodePointIndex]); // can we exclude the PP self interactions
                                                                        // somehow in the for loops?
                            }
                        }
                    }
                }
            }
            else
            {
                for (int firstNodePointIndex = node_A.firstParticleIndex;
                     firstNodePointIndex < node_A.firstParticleIndex + node_A.particleIndexAmount;
                     firstNodePointIndex++)
                {
                    if (node_B.particleIndexAmount != 0)
                    {
                        traverse_SYM_PN(total, points, points[firstNodePointIndex], node_B, theta);
                    }
                }
            }
        }
        else if (node_B.firstChildIndex == 0) // traverse for each point in second
        {

            if (&node_A == &node_B)
            {
                for (int secondNodePointIndex = node_B.firstParticleIndex;
                     secondNodePointIndex < node_B.firstParticleIndex + node_B.particleIndexAmount;
                     secondNodePointIndex++)
                {
                    if (node_A.particleIndexAmount != 0)
                    {
                        for (int firstNodePointIndex = secondNodePointIndex;
                             firstNodePointIndex < node_A.firstParticleIndex + node_A.particleIndexAmount;
                             firstNodePointIndex++)
                        {
                            if (points[firstNodePointIndex].ID != points[secondNodePointIndex].ID)
                            {

                                kernelPP(total,
                                         points[firstNodePointIndex],
                                         points[secondNodePointIndex]); // can we exclude the PP self interactions
                                                                        // somehow in the for loops?
                            }
                        }
                    }
                }
            }
            else
            {
                for (int secondNodePointIndex = node_B.firstParticleIndex;
                     secondNodePointIndex < node_B.firstParticleIndex + node_B.particleIndexAmount;
                     secondNodePointIndex++)
                {
                    if (node_A.particleIndexAmount != 0)
                    {
                        traverse_SYM_PN(total, points, points[secondNodePointIndex], node_A, theta);
                    }
                }
            }
        }
        else // traverse for each node in first and second
        {

            if (&node_A == &node_B) // if nodes from interaction pair are the same then dont add duplicates
            {
                for (int firstNodeChildIndex = node_A.firstChildIndex; firstNodeChildIndex < node_A.firstChildIndex + 4;
                     firstNodeChildIndex++)
                {
                    for (int secondNodeChildIndex = firstNodeChildIndex;
                         secondNodeChildIndex < node_A.firstChildIndex + 4;
                         secondNodeChildIndex++)
                    {

                        if (nodes[firstNodeChildIndex].particleIndexAmount != 0 &&
                            nodes[secondNodeChildIndex].particleIndexAmount != 0)
                            traverse_SYM_NN(
                                total, points, nodes[firstNodeChildIndex], nodes[secondNodeChildIndex], theta);
                    }
                }
            }
            else // if nodes from interaction pair are the NOT same then add all combinations
            {
                for (int firstNodeChildIndex = node_A.firstChildIndex; firstNodeChildIndex < node_A.firstChildIndex + 4;
                     firstNodeChildIndex++)
                {
                    for (int secondNodeChildIndex = node_B.firstChildIndex;
                         secondNodeChildIndex < node_B.firstChildIndex + 4;
                         secondNodeChildIndex++)
                    {

                        if (nodes[firstNodeChildIndex].particleIndexAmount != 0 &&
                            nodes[secondNodeChildIndex].particleIndexAmount != 0)
                            traverse_SYM_NN(
                                total, points, nodes[firstNodeChildIndex], nodes[secondNodeChildIndex], theta);
                    }
                }
            }
        }
    }

    void
    traverse_SYM_PN(double& total, std::vector<T>& points, T& firstPoint, NodeFMM_MORTON_2D& secondNode, double theta)
    {
        glm::dvec2 diff = firstPoint.position - secondNode.centreOfMass;
        // double dist = glm::length(diff);
        double dist = glm::dot(diff, diff);

        // double threshold = (secondNode.BBlength) / (theta * 0.35);
        double threshold = secondNode.BBlength / theta;
        if (dist > threshold * threshold) // sym point node
        // if (secondNode.BBlength / dist < theta)
        {

            kernelPN(total, firstPoint, secondNode);
        }
        else if (secondNode.firstChildIndex == 0) // sym point point
        {

            for (int secondNodePointIndex = secondNode.firstParticleIndex;
                 secondNodePointIndex < secondNode.firstParticleIndex + secondNode.particleIndexAmount;
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

            for (int secondNodeChildIndex = secondNode.firstChildIndex;
                 secondNodeChildIndex < secondNode.firstChildIndex + 4;
                 secondNodeChildIndex++)
            {

                if (nodes[secondNodeChildIndex].particleIndexAmount != 0)
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

    void createLeafNodes(std::vector<T>& points, glm::dvec2 minPos, glm::dvec2 maxPos)
    {
        glm::dvec2 negMinPos = -minPos;
        double largestAxis = glm::compMax(maxPos + negMinPos);

        double leafNodeSize = largestAxis / static_cast<double>(levelGridWidth[treeDepth]);

        for (int i = 0; i < points.size(); i++)
        {
            glm::dvec2 gridPos =
                static_cast<double>(levelGridWidth[treeDepth]) * (points[i].position + negMinPos) / (largestAxis);

            glm::vec<2, uint32_t> gridCoord =
                glm::min(glm::max(glm::vec<2, uint32_t>(gridPos), glm::vec<2, uint32_t>(0u)),
                         glm::vec<2, uint32_t>(levelGridWidth[treeDepth] - 1u));

            uint32_t leafLevelIndex =
                (uint32_t) levelIndex[treeDepth] + ((spread16(gridCoord.y) << 1) | spread16(gridCoord.x));

            if (nodes[leafLevelIndex].particleIndexAmount == 0u)
            {
                nodes[leafLevelIndex].BBcentre =
                    minPos + leafNodeSize * glm::dvec2(gridCoord) + 0.5 * glm::dvec2(leafNodeSize);
                nodes[leafLevelIndex].BBlength = leafNodeSize;

                nodes[leafLevelIndex].firstParticleIndex = i;
            }

            nodes[leafLevelIndex].particleIndexAmount += 1u;
            nodes[leafLevelIndex].centreOfMass += points[i].position;
            // nodes[leafLevelIndex].M0 += 1.0;
        }

        for (int n = levelIndex[treeDepth]; n < levelIndex[treeDepth] + levelSize[treeDepth]; n++)
        {
            // if (nodes[n].M0 != 0.0)
            if (nodes[n].particleIndexAmount != 0u)
            {
                nodes[n].M0 = nodes[n].particleIndexAmount;
                nodes[n].centreOfMass /= nodes[n].M0;

                for (int pointIndex = nodes[n].firstParticleIndex;
                     pointIndex < nodes[n].firstParticleIndex + nodes[n].particleIndexAmount;
                     pointIndex++)
                {
                    glm::dvec2 relativeCoord = points[pointIndex].position - nodes[n].centreOfMass;

                    Fastor::Tensor<double, 2, 2> outer_product;
                    outer_product(0, 0) = relativeCoord.x * relativeCoord.x;
                    outer_product(0, 1) = relativeCoord.x * relativeCoord.y;
                    outer_product(1, 0) = relativeCoord.y * relativeCoord.x;
                    outer_product(1, 1) = relativeCoord.y * relativeCoord.y;
                    nodes[n].M2 += outer_product;
                }
            }
        }
    }

    void bottomUpNodeConstruction(glm::dvec2 minPos, glm::dvec2 maxPos)
    {
        double max_minus_min_pos = maxPos.x - minPos.x;

        std::array<glm::dvec2, 4> BBcentreOffset{
            glm::dvec2(0.5, 0.5), glm::dvec2(-0.5, 0.5), glm::dvec2(0.5, -0.5), glm::dvec2(-0.5, -0.5)};

        for (int l = treeDepth - 1; l >= 0; l--)
        {
            // const int cells_per_axis = 1 << l;
            // const int cells_per_axis_child = 1 << (l+1);
            // const double BBlength_l = max_minus_min_pos / static_cast<double>(cells_per_axis);
            // const double BBlength_l_child = max_minus_min_pos / static_cast<double>(cells_per_axis_child);

            for (int i = 0; i < levelSize[l]; i++)
            {
                int nodeIndex = levelIndex[l] + i;
                unsigned int potentialFirstChildIndex = levelIndex[l + 1] + i * 4;

                // unsigned int children_counter = 0u;
                // unsigned int only_child_index = 0u;

                // nodes[nodeIndex].BBlength = BBlength_l;

                // uint32_t ix = compact16(i);
                // uint32_t iy = compact16(i >> 1);
                // nodes[nodeIndex].BBcentre = minPos + glm::dvec2(ix + 0.5, iy + 0.5) * BBlength_l;

                for (int j = 3; j >= 0; j--) // loop over all children of node i in level l
                {
                    unsigned int childIndex = potentialFirstChildIndex + j;
                    if (nodes[childIndex].particleIndexAmount != 0u)
                    {
                        // children_counter++;
                        // only_child_index = childIndex;

                        nodes[nodeIndex].firstChildIndex = potentialFirstChildIndex;

                        nodes[nodeIndex].BBcentre =
                            BBcentreOffset[j] * nodes[childIndex].BBlength + nodes[childIndex].BBcentre; // old way
                        nodes[nodeIndex].BBlength = 2.0 * nodes[childIndex].BBlength;                    // old way

                        nodes[nodeIndex].firstParticleIndex = nodes[childIndex].firstParticleIndex;
                        nodes[nodeIndex].particleIndexAmount += nodes[childIndex].particleIndexAmount;

                        nodes[nodeIndex].centreOfMass += nodes[childIndex].M0 * nodes[childIndex].centreOfMass;

                        nodes[nodeIndex].M0 += nodes[childIndex].M0;
                    }
                }

                // if (children_counter == 1u)
                //{
                //     nodes[nodeIndex].BBcentre = nodes[only_child_index].BBcentre;
                //     nodes[nodeIndex].BBlength = nodes[only_child_index].BBlength;

                //    nodes[nodeIndex].centreOfMass = nodes[only_child_index].centreOfMass;
                //    nodes[nodeIndex].M2 = nodes[only_child_index].M2;

                //    continue;
                //}

                if (nodes[nodeIndex].particleIndexAmount != 0u)
                {
                    nodes[nodeIndex].centreOfMass /= nodes[nodeIndex].M0;

                    for (int nodeChildIndex = nodes[nodeIndex].firstChildIndex;
                         nodeChildIndex < nodes[nodeIndex].firstChildIndex + 4;
                         nodeChildIndex++)
                    {
                        if (nodes[nodeChildIndex].M0 != 0.0)
                        {
                            glm::dvec2 relativeCoord =
                                nodes[nodeChildIndex].centreOfMass - nodes[nodeIndex].centreOfMass;

                            Fastor::Tensor<double, 2, 2> outer_product;
                            outer_product(0, 0) = relativeCoord.x * relativeCoord.x;
                            outer_product(0, 1) = relativeCoord.x * relativeCoord.y;
                            outer_product(1, 0) = relativeCoord.y * relativeCoord.x;
                            outer_product(1, 1) = relativeCoord.y * relativeCoord.y;
                            nodes[nodeIndex].M2 += outer_product;

                            nodes[nodeIndex].M2 += nodes[nodeChildIndex].M2;
                        }
                    }
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

    void applyForces(std::vector<T>& points, NodeFMM_MORTON_2D& node)
    {
        if (node.firstChildIndex != 0)
        {
            for (int nodeIndex = node.firstChildIndex; nodeIndex < node.firstChildIndex + 4; nodeIndex++)
            {
                if (nodes[nodeIndex].particleIndexAmount != 0)
                {
                    double one_over_M0 = 1.0 / nodes[nodeIndex].M0;
                    nodes[nodeIndex].C2 *= one_over_M0;
                    nodes[nodeIndex].C3 *= one_over_M0;

                    glm::dvec2 oldZ = nodes[nodeIndex].centreOfMass;
                    glm::dvec2 newZ = node.centreOfMass;
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
        }
        else
        {
            for (int pointIndex = node.firstParticleIndex;
                 pointIndex < node.firstParticleIndex + node.particleIndexAmount;
                 pointIndex++)
            {
                glm::dvec2 x = points[pointIndex].position;
                glm::dvec2 Z0 = node.centreOfMass;
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

    void
    getNodesBufferData(std::vector<VertexPos2Col3>& nodesBufferData, NodeFMM_MORTON_2D node, int level, int showLevel)
    {
        if ((level == showLevel || showLevel == -1) && node.particleIndexAmount != 0)
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

            // const int cells_per_axis = 1 << level;
            // const double BBlength_l = max_minus_min_pos / static_cast<double>(cells_per_axis);

            glm::vec2 lowestCorner = node.BBcentre - 0.5f * node.BBlength;
            glm::vec2 highestCorner = node.BBcentre + 0.5f * node.BBlength;

            nodesBufferData.push_back(VertexPos2Col3(glm::vec2(lowestCorner.x, lowestCorner.y), color));
            nodesBufferData.push_back(VertexPos2Col3(glm::vec2(highestCorner.x, lowestCorner.y), color));

            nodesBufferData.push_back(VertexPos2Col3(glm::vec2(lowestCorner.x, lowestCorner.y), color));
            nodesBufferData.push_back(VertexPos2Col3(glm::vec2(lowestCorner.x, highestCorner.y), color));

            nodesBufferData.push_back(VertexPos2Col3(glm::vec2(lowestCorner.x, highestCorner.y), color));
            nodesBufferData.push_back(VertexPos2Col3(glm::vec2(highestCorner.x, highestCorner.y), color));

            nodesBufferData.push_back(VertexPos2Col3(glm::vec2(highestCorner.x, lowestCorner.y), color));
            nodesBufferData.push_back(VertexPos2Col3(glm::vec2(highestCorner.x, highestCorner.y), color));

            // float crossSize = 0.1f;
            // nodesBufferData.push_back(VertexPos2Col3(node.centreOfMass - glm::vec2(crossSize, 0.0f),
            // glm::vec3(1.0f,0.0f,0.0f))); nodesBufferData.push_back(VertexPos2Col3(node.centreOfMass +
            // glm::vec2(crossSize, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f)));

            // nodesBufferData.push_back(VertexPos2Col3(node.centreOfMass - glm::vec2(0.0f, crossSize), glm::vec3(1.0f,
            // 0.0f, 0.0f))); nodesBufferData.push_back(VertexPos2Col3(node.centreOfMass + glm::vec2(0.0f, crossSize),
            // glm::vec3(1.0f, 0.0f, 0.0f)));
        }

        if (node.firstChildIndex != 0u)
        {
            for (int i = node.firstChildIndex; i < node.firstChildIndex + 4; i++)
            {
                getNodesBufferData(nodesBufferData, nodes[i], level + 1, showLevel);
            }
        }
    }
};

// TSNE kernals
// -------------------------------------------------------------
inline void TSNE_FMM_SYM_MORTON_NN_Kernel(double& total, NodeFMM_MORTON_2D& sinkNode, NodeFMM_MORTON_2D& sourceNode)
{
    glm::dvec2 R = sinkNode.centreOfMass - sourceNode.centreOfMass;
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

inline void TSNE_FMM_SYM_MORTON_PN_Kernel(double& total, TsnePoint2D& sinkPoint, NodeFMM_MORTON_2D& sourceNode)
{
    glm::dvec2 R = sinkPoint.position - sourceNode.centreOfMass;
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

inline void TSNE_FMM_SYM_MORTON_PP_Kernel(double& total, TsnePoint2D& sinkPoint, TsnePoint2D& sourcePoint)
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

/*
#pragma once

#include <functional>
#include <glm/glm.hpp>
#include <glm/gtx/component_wise.hpp>
#include <glm/gtx/string_cast.hpp>
#include <utility>
#include <vector>
#include <Fastor/Fastor.h>
#include <boost/sort/sort.hpp>
#include <cstdint>
#include <algorithm>
#include <string>
#include <exception>
#include <thread>
#include <chrono>

#include "../../common.h"
#include "nBodySolver.h"
#include "../../trees/cpu/quadtreeFMM.h"
#include "../../trees/cpu/nodeFMM_MORTON_2D.h"
#include "../../particles/embeddedPoint.h"
#include "../../particles/tsnePoint2D.h"
#include "../../particles/Particle2D.h"
#include "../../particles/morton_point.h"



template <typename T>
class NBodySolverFMM_SYM_MORTON : public NBodySolver<T>
{
public:
    unsigned int treeDepth;
    std::vector<NodeFMM_MORTON_2D> nodes;
    std::vector<unsigned int> levelIndex;
    std::vector<unsigned int> levelSize;
    std::vector<unsigned int> levelGridWidth;

    std::function<void(double&, NodeFMM_MORTON_2D&, NodeFMM_MORTON_2D&)> kernelNN;
    std::function<void(double&, T&, NodeFMM_MORTON_2D&)> kernelPN;
    //std::function<void(double&, NodeFMM_MORTON_2D&, T&)> kernelNP;
    std::function<void(double&, T&, T&)> kernelPP;

    std::vector<std::pair<NodeFMM_MORTON_2D, NodeFMM_MORTON_2D>> interaction_NN_stack;
    std::vector<std::pair<T, NodeFMM_MORTON_2D>> interaction_PN_stack;

    NBodySolverFMM_SYM_MORTON() {}

    NBodySolverFMM_SYM_MORTON
    (
        std::function<void(double&, NodeFMM_MORTON_2D&, NodeFMM_MORTON_2D&)> initKernelNN,
        std::function<void(double&, T&, NodeFMM_MORTON_2D&)> initKernelPN,
        //std::function<void(double&, NodeFMM_MORTON_2D&, T&)> initKernelNP,
        std::function<void(double&, T&, T&)> initKernelPP,
        int initMaxChildren,
        unsigned int initTreeDepth,
        float initTheta
    )
    {
        kernelNN = initKernelNN;
        kernelPN = initKernelPN;
        //kernelNP = initKernelNP;
        kernelPP = initKernelPP;
        this->maxChildren = initMaxChildren;
        initNodesSize(initTreeDepth);
        this->theta = initTheta;
    }

    void solveNbody(double& total, std::vector<T>& points, std::vector<int>& indexTracker) override
    {
        traverse_SYM_NN(total, points, nodes[0], nodes[0], this->theta);
        divide_by_mass();
        applyForces(points, nodes[0]);
    }

    void updateTree(std::vector<T>& points, glm::vec2 minPos, glm::vec2 maxPos) override
    {
        NodeFMM_MORTON_2D emptyNode;
        std::fill(nodes.begin(), nodes.end(), emptyNode);

        glm::vec2 negMinPos = -minPos;
        float largestAxis = glm::compMax(maxPos + negMinPos);


        std::vector<MortonPoint<T>> pointsMortons(points.size());
        for (int i = 0; i < points.size(); i++)
            pointsMortons[i] = MortonPoint<T>(createMortonCode(points[i].position, negMinPos, largestAxis), points[i]);
        boost::sort::spreadsort::integer_sort(pointsMortons.begin(), pointsMortons.end(), rightshift<T>(),
lessthan<T>()); for (int i = 0; i < points.size(); i++) points[i] = pointsMortons[i].point;


        createLeafNodes(points, minPos, maxPos);

        bottomUpNodeConstruction();
    }

    std::vector<VertexPos2Col3> getNodesBufferData(int nodeLevelToShow) override
    {
        std::vector<VertexPos2Col3> result;

        getNodesBufferData(result, nodes[0], 0, nodeLevelToShow);

        return result;
    }

    static unsigned int getDepth(int max_children, int N)
    {
        unsigned int depth = 0;
        while (N > max_children)
        {
            N /= 4;
            depth++;
        }

        return depth;
    }

private:
    void traverse_SYM_NN(double& total, std::vector<T>& points, NodeFMM_MORTON_2D& node_A, NodeFMM_MORTON_2D& node_B,
float theta)
    {
        //interaction_NN_stack.clear();
        //interaction_NN_stack.push_back(std::pair<NodeFMM_MORTON_2D, NodeFMM_MORTON_2D>{node_A, node_B});

        //while (interaction_NN_stack.size() != 0)
        //{
        //    std::pair<NodeFMM_MORTON_2D, NodeFMM_MORTON_2D> cur_pair = interaction_NN_stack.back();
        //    interaction_NN_stack.pop_back();
        std::pair<NodeFMM_MORTON_2D&, NodeFMM_MORTON_2D&> cur_pair = std::pair<NodeFMM_MORTON_2D&, NodeFMM_MORTON_2D&>{
node_A, node_B };


        glm::vec2 diff = cur_pair.first.centreOfMass - cur_pair.second.centreOfMass;
        float dist = glm::length(diff);

        if ((cur_pair.first.BBlength + cur_pair.second.BBlength) / dist < theta) // sym node node
        {
            if (cur_pair.first.particleIndexAmount != 0 && cur_pair.second.particleIndexAmount != 0)
            {

                kernelNN(total, cur_pair.first, cur_pair.second);

            }
        }
        else if (cur_pair.first.firstChildIndex == 0) // traverse for each point in first
        {
            for (int firstNodePointIndex = cur_pair.first.firstParticleIndex; firstNodePointIndex <
cur_pair.first.firstParticleIndex + cur_pair.first.particleIndexAmount; firstNodePointIndex++)
            {

                if (cur_pair.second.particleIndexAmount != 0)
                {
                    if (&cur_pair.first == &cur_pair.second)
                    {
                        for (int secondNodePointIndex = firstNodePointIndex; secondNodePointIndex <
cur_pair.second.firstParticleIndex + cur_pair.second.particleIndexAmount; secondNodePointIndex++)
                        {
                            if (points[firstNodePointIndex].ID != points[secondNodePointIndex].ID)
                            {

                                kernelPP(total, points[firstNodePointIndex], points[secondNodePointIndex]); // can we
exclude the PP self interactions somehow in the for loops?

                            }
                        }
                    }
                    else
                    {
                        traverse_SYM_PN(total, points, points[firstNodePointIndex], cur_pair.second, theta);
                    }
                }

            }
        }
        else if (cur_pair.second.firstChildIndex == 0) // traverse for each point in second
        {
            for (int secondNodePointIndex = cur_pair.second.firstParticleIndex; secondNodePointIndex <
cur_pair.second.firstParticleIndex + cur_pair.second.particleIndexAmount; secondNodePointIndex++)
            {

                if (cur_pair.first.particleIndexAmount != 0)
                {
                    if (&cur_pair.first == &cur_pair.second)
                    {
                        for (int firstNodePointIndex = secondNodePointIndex; firstNodePointIndex <
cur_pair.first.firstParticleIndex + cur_pair.first.particleIndexAmount; firstNodePointIndex++)
                        {
                            if (points[firstNodePointIndex].ID != points[secondNodePointIndex].ID)
                            {

                                kernelPP(total, points[firstNodePointIndex], points[secondNodePointIndex]); // can we
exclude the PP self interactions somehow in the for loops?

                            }
                        }
                    }
                    else
                    {
                        traverse_SYM_PN(total, points, points[secondNodePointIndex], cur_pair.first, theta);
                    }
                }
            }
        }
        else // traverse for each node in first and second
        {
            if (&cur_pair.first == &cur_pair.second) // if nodes from interaction pair are the same then dont add
duplicates
            {
                for (int firstNodeChildIndex = cur_pair.first.firstChildIndex; firstNodeChildIndex <
cur_pair.first.firstChildIndex + 4; firstNodeChildIndex++)
                {
                    for (int secondNodeChildIndex = firstNodeChildIndex; secondNodeChildIndex <
cur_pair.first.firstChildIndex + 4; secondNodeChildIndex++)
                    {

                        if (nodes[firstNodeChildIndex].particleIndexAmount != 0 &&
nodes[secondNodeChildIndex].particleIndexAmount != 0) traverse_SYM_NN(total, points, nodes[firstNodeChildIndex],
nodes[secondNodeChildIndex], theta);

                    }
                }
            }
            else // if nodes from interaction pair are the NOT same then add all combinations
            {
                for (int firstNodeChildIndex = cur_pair.first.firstChildIndex; firstNodeChildIndex <
cur_pair.first.firstChildIndex + 4; firstNodeChildIndex++)
                {
                    for (int secondNodeChildIndex = cur_pair.second.firstChildIndex; secondNodeChildIndex <
cur_pair.second.firstChildIndex + 4; secondNodeChildIndex++)
                    {

                        if (nodes[firstNodeChildIndex].particleIndexAmount != 0 &&
nodes[secondNodeChildIndex].particleIndexAmount != 0) traverse_SYM_NN(total, points, nodes[firstNodeChildIndex],
nodes[secondNodeChildIndex], theta);

                    }
                }
            }
        }

        //}
    }

    void traverse_SYM_PN(double& total, std::vector<T>& points, T& firstPoint, NodeFMM_MORTON_2D& secondNode, float
theta)
    {
        //interaction_PN_stack.clear();
        //interaction_PN_stack.push_back(std::pair<T, NodeFMM_MORTON_2D>{firstPoint, secondNode});

        //while (interaction_PN_stack.size() != 0)
        //{
        //    std::pair<T, NodeFMM_MORTON_2D> cur_pair = interaction_PN_stack.back();
        //    interaction_PN_stack.pop_back();

        std::pair<T&, NodeFMM_MORTON_2D&> cur_pair = std::pair<T&, NodeFMM_MORTON_2D&>{ firstPoint, secondNode };

        glm::vec2 diff = cur_pair.first.position - cur_pair.second.centreOfMass;
        float dist = glm::length(diff);

        if (cur_pair.second.BBlength / dist < theta * 0.5f) // sym point node
        {

            kernelPN(total, cur_pair.first, cur_pair.second);

        }
        else if (cur_pair.second.firstChildIndex == 0) // sym point point
        {
            for (int secondNodePointIndex = cur_pair.second.firstParticleIndex; secondNodePointIndex <
cur_pair.second.firstParticleIndex + cur_pair.second.particleIndexAmount; secondNodePointIndex++)
            {
                if (points[secondNodePointIndex].ID != cur_pair.first.ID)
                {

                    kernelPP(total, cur_pair.first, points[secondNodePointIndex]);

                }
            }
        }
        else // traverse for each node in second
        {
            for (int secondNodeChildIndex = cur_pair.second.firstChildIndex; secondNodeChildIndex <
cur_pair.second.firstChildIndex + 4; secondNodeChildIndex++)
            {

                if (nodes[secondNodeChildIndex].particleIndexAmount != 0)
                    traverse_SYM_PN(total, points, cur_pair.first, nodes[secondNodeChildIndex], theta);

            }
        }
        //}
    }

    void divide_by_mass()
    {
        for (int i = 0; i < nodes.size(); i++)
        {
            if (nodes[i].M0 != 0.0f)
            {
                nodes[i].C1 /= nodes[i].M0;
                nodes[i].C2 /= nodes[i].M0;
                nodes[i].C3 /= nodes[i].M0;
            }
        }
    }

    void initNodesSize(unsigned int initTreeDepth)
    {
        treeDepth = initTreeDepth;
        levelIndex.resize(treeDepth + 1);
        levelSize.resize(treeDepth + 1);
        levelGridWidth.resize(treeDepth + 1);
        levelIndex[0] = 0;
        int nodesSize = 0;
        for (int i = 0; i <= treeDepth; i++) // treeDepth = 0 means just the root
        {
            int currentLevelSize = 1;
            for (int j = 0; j < i; j++)
            {
                currentLevelSize *= 4;
            }
            levelSize[i] = currentLevelSize;
            nodesSize += currentLevelSize;

            int currentDepthStart = 0;
            for (int j = 0; j <= i; j++)
            {
                currentDepthStart += levelSize[j];
            }
            if (i + 1 < treeDepth + 1)
                levelIndex[i + 1] = currentDepthStart;


        }

        nodes.resize(nodesSize);

        for (int i = 0; i <= treeDepth; i++)
            levelGridWidth[i] = std::pow(2, i);
        //leafGridSize = std::pow(2, treeDepth);
    }

    void createLeafNodes(std::vector<T>& points, glm::vec2 minPos, glm::vec2 maxPos)
    {
        glm::vec2 negMinPos = -minPos;
        float largestAxis = glm::compMax(maxPos + negMinPos);

        float leafNodeSize = largestAxis / static_cast<float>(levelGridWidth[treeDepth]);

        for (int i = 0; i < points.size(); i++)
        {
            glm::vec2 gridPos = static_cast<float>(levelGridWidth[treeDepth]) * (points[i].position + negMinPos) /
(largestAxis);

            glm::vec<2, uint32_t> gridCoord = glm::min(glm::max(glm::vec<2, uint32_t>(gridPos), glm::vec<2,
uint32_t>(0u)), glm::vec<2, uint32_t>(levelGridWidth[treeDepth] - 1u));

            uint32_t leafLevelIndex = (uint32_t)levelIndex[treeDepth] + ((spread16(gridCoord.y) << 1) |
spread16(gridCoord.x));

            if (nodes[leafLevelIndex].particleIndexAmount == 0u)
            {
                nodes[leafLevelIndex].BBcentre = minPos + leafNodeSize * glm::vec2(gridCoord) + 0.5f *
glm::vec2(leafNodeSize); nodes[leafLevelIndex].BBlength = leafNodeSize;

                nodes[leafLevelIndex].firstParticleIndex = i;
            }

            nodes[leafLevelIndex].particleIndexAmount += 1u;
            nodes[leafLevelIndex].centreOfMass += points[i].position;
            nodes[leafLevelIndex].M0 += 1.0f;
        }

        for (int n = levelIndex[treeDepth]; n < levelIndex[treeDepth] + levelSize[treeDepth]; n++)
        {
            if (nodes[n].M0 != 0.0f)
            {
                nodes[n].centreOfMass /= nodes[n].M0;

                for (int pointIndex = nodes[n].firstParticleIndex; pointIndex < nodes[n].firstParticleIndex +
nodes[n].particleIndexAmount; pointIndex++)
                {
                    glm::vec2 relativeCoord = points[pointIndex].position - nodes[n].centreOfMass;

                    Fastor::Tensor<float, 2, 2> outer_product;
                    outer_product(0, 0) = relativeCoord.x * relativeCoord.x;
                    outer_product(0, 1) = relativeCoord.x * relativeCoord.y;
                    outer_product(1, 0) = relativeCoord.y * relativeCoord.x;
                    outer_product(1, 1) = relativeCoord.y * relativeCoord.y;
                    nodes[n].M2 += outer_product;
                }
            }
        }


    }

    void bottomUpNodeConstruction()
    {
        std::array<glm::vec2, 4> BBcentreOffset
        {
            glm::vec2(0.5f, 0.5f),
            glm::vec2(-0.5f, 0.5f),
            glm::vec2(0.5f, -0.5f),
            glm::vec2(-0.5f, -0.5f)
        };

        for (int l = treeDepth - 1; l >= 0; l--)
        {
            for (int i = 0; i < levelSize[l]; i++)
            {
                int nodeIndex = levelIndex[l] + i;
                unsigned int potentialFirstChildIndex = levelIndex[l + 1] + i * 4;

                for (int j = 3; j >= 0; j--) // loop over all children of node i in level l
                {
                    unsigned int childIndex = potentialFirstChildIndex + j;
                    if (nodes[childIndex].particleIndexAmount != 0u)
                    {
                        nodes[nodeIndex].firstChildIndex = potentialFirstChildIndex;

                        nodes[nodeIndex].BBcentre = BBcentreOffset[j] * nodes[childIndex].BBlength +
nodes[childIndex].BBcentre; nodes[nodeIndex].BBlength = 2.0f * nodes[childIndex].BBlength;

                        nodes[nodeIndex].firstParticleIndex = nodes[childIndex].firstParticleIndex;
                        nodes[nodeIndex].particleIndexAmount += nodes[childIndex].particleIndexAmount;

                        nodes[nodeIndex].centreOfMass += nodes[childIndex].M0 * nodes[childIndex].centreOfMass;

                        nodes[nodeIndex].M0 += nodes[childIndex].M0;
                    }
                }

                if (nodes[nodeIndex].particleIndexAmount != 0)
                {
                    nodes[nodeIndex].centreOfMass /= nodes[nodeIndex].M0;

                    for (int nodeChildIndex = nodes[nodeIndex].firstChildIndex; nodeChildIndex <
nodes[nodeIndex].firstChildIndex + 4; nodeChildIndex++)
                    {
                        if (nodes[nodeChildIndex].M0 != 0.0f)
                        {
                            glm::vec2 relativeCoord = nodes[nodeChildIndex].centreOfMass -
nodes[nodeIndex].centreOfMass;

                            Fastor::Tensor<float, 2, 2> outer_product;
                            outer_product(0, 0) = relativeCoord.x * relativeCoord.x;
                            outer_product(0, 1) = relativeCoord.x * relativeCoord.y;
                            outer_product(1, 0) = relativeCoord.y * relativeCoord.x;
                            outer_product(1, 1) = relativeCoord.y * relativeCoord.y;
                            nodes[nodeIndex].M2 += outer_product;

                            nodes[nodeIndex].M2 += nodes[nodeChildIndex].M2;
                        }
                    }
                }
            }
        }
    }

    void applyForces(std::vector<T>& points, NodeFMM_MORTON_2D& node)
    {
        if (node.firstChildIndex != 0)
        {
            for (int nodeIndex = node.firstChildIndex; nodeIndex < node.firstChildIndex + 4; nodeIndex++)
            {
                if (nodes[nodeIndex].particleIndexAmount != 0)
                {
                    glm::vec2 oldZ = nodes[nodeIndex].centreOfMass;
                    glm::vec2 newZ = node.centreOfMass;
                    Fastor::Tensor<float, 2> diff1 = { oldZ.x - newZ.x, oldZ.y - newZ.y };
                    Fastor::Tensor<float, 2, 2> diff2 = Fastor::outer(diff1, diff1);
                    Fastor::Tensor<float, 2, 2, 2> diff3 = Fastor::outer(diff2, diff1);

                    Fastor::Tensor<float, 2> newC1 = node.C1 +
                        Fastor::einsum<Fastor::Index<0>, Fastor::Index<0, 1>>(diff1, node.C2) +
                        (1.0f / 2.0f) * Fastor::einsum<Fastor::Index<0>, Fastor::Index<0, 1>>(diff1,
Fastor::einsum<Fastor::Index<0>, Fastor::Index<0, 1, 2>>(diff1, node.C3));

                    Fastor::Tensor<float, 2, 2> newC2 = node.C2 +
                        Fastor::einsum<Fastor::Index<0>, Fastor::Index<0, 1, 2>>(diff1, node.C3);

                    Fastor::Tensor<float, 2, 2, 2> newC3 = node.C3;

                    nodes[nodeIndex].C1 += newC1;
                    nodes[nodeIndex].C2 += newC2;
                    nodes[nodeIndex].C3 += newC3;

                    applyForces(points, nodes[nodeIndex]);
                }
            }
        }
        else
        {
            for (int pointIndex = node.firstParticleIndex; pointIndex < node.firstParticleIndex +
node.particleIndexAmount; pointIndex++)
            {
                glm::vec2 x = points[pointIndex].position;
                glm::vec2 Z0 = node.centreOfMass;
                Fastor::Tensor<float, 2> diff1 = { x.x - Z0.x, x.y - Z0.y };
                Fastor::Tensor<float, 2, 2> diff2 = Fastor::outer(diff1, diff1);
                Fastor::Tensor<float, 2, 2, 2> diff3 = Fastor::outer(diff2, diff1);

                Fastor::Tensor<float, 2> newC1 = node.C1 +
                    Fastor::einsum<Fastor::Index<0>, Fastor::Index<0, 1>>(diff1, node.C2) +
                    (1.0f / 2.0f) * Fastor::einsum<Fastor::Index<0>, Fastor::Index<0, 1>>(diff1,
Fastor::einsum<Fastor::Index<0>, Fastor::Index<0, 1, 2>>(diff1, node.C3));

                points[pointIndex].derivative += glm::vec2(newC1(0), newC1(1));
            }
        }
    }

    uint32_t createMortonCode(glm::vec2 position, glm::vec2 negMinPos, float largestAxis)
    {
        position = (position + negMinPos) / (largestAxis); // rescale such that range is between [0-1]

        position = glm::min(glm::max(position * 65536.0f, glm::vec2(0.0f)), glm::vec2(65535.0f)); // rescale such that
range is between [0-65535] which is the max number for 16 bits

        return (spread16((uint32_t)position.y) << 1) | spread16((uint32_t)position.x);
    }

    inline uint32_t spread16(uint32_t x)
    {
        x = (x | (x << 8)) & 0x00FF00FFu;
        x = (x | (x << 4)) & 0x0F0F0F0Fu;
        x = (x | (x << 2)) & 0x33333333u;
        x = (x | (x << 1)) & 0x55555555u;
        return x;
    }

    void getNodesBufferData(std::vector<VertexPos2Col3>& nodesBufferData, NodeFMM_MORTON_2D node, int level, int
showLevel)
    {
        if ((level == showLevel || showLevel == -1) && node.particleIndexAmount != 0)
        {
            const int colorsSize = 7;
            std::array<glm::vec3, colorsSize> colors{
                glm::vec3(1.0f, 1.0f, 1.0f),
                glm::vec3(1.0f, 0.0f, 0.0f),
                glm::vec3(0.0f, 1.0f, 0.0f),
                glm::vec3(0.0f, 0.0f, 1.0f),
                glm::vec3(1.0f, 1.0f, 0.0f),
                glm::vec3(0.0f, 1.0f, 1.0f),
                glm::vec3(1.0f, 0.0f, 1.0f)
            };

            glm::vec3 color = colors[std::min(showLevel + 1, colorsSize - 1)];

            glm::vec2 lowestCorner = node.BBcentre - 0.5f * node.BBlength;
            glm::vec2 highestCorner = node.BBcentre + 0.5f * node.BBlength;

            nodesBufferData.push_back(VertexPos2Col3(glm::vec2(lowestCorner.x, lowestCorner.y), color));
            nodesBufferData.push_back(VertexPos2Col3(glm::vec2(highestCorner.x, lowestCorner.y), color));

            nodesBufferData.push_back(VertexPos2Col3(glm::vec2(lowestCorner.x, lowestCorner.y), color));
            nodesBufferData.push_back(VertexPos2Col3(glm::vec2(lowestCorner.x, highestCorner.y), color));

            nodesBufferData.push_back(VertexPos2Col3(glm::vec2(lowestCorner.x, highestCorner.y), color));
            nodesBufferData.push_back(VertexPos2Col3(glm::vec2(highestCorner.x, highestCorner.y), color));

            nodesBufferData.push_back(VertexPos2Col3(glm::vec2(highestCorner.x, lowestCorner.y), color));
            nodesBufferData.push_back(VertexPos2Col3(glm::vec2(highestCorner.x, highestCorner.y), color));

            //float crossSize = 0.1f;
            //nodesBufferData.push_back(VertexPos2Col3(node.centreOfMass - glm::vec2(crossSize, 0.0f),
glm::vec3(1.0f,0.0f,0.0f)));
            //nodesBufferData.push_back(VertexPos2Col3(node.centreOfMass + glm::vec2(crossSize, 0.0f), glm::vec3(1.0f,
0.0f, 0.0f)));

            //nodesBufferData.push_back(VertexPos2Col3(node.centreOfMass - glm::vec2(0.0f, crossSize), glm::vec3(1.0f,
0.0f, 0.0f)));
            //nodesBufferData.push_back(VertexPos2Col3(node.centreOfMass + glm::vec2(0.0f, crossSize), glm::vec3(1.0f,
0.0f, 0.0f)));
        }

        if (node.firstChildIndex != 0)
        {
            for (int i = node.firstChildIndex; i < node.firstChildIndex + 4; i++)
            {
                getNodesBufferData(nodesBufferData, nodes[i], level + 1, showLevel);
            }
        }
    }
};



// TSNE kernals
----------------------------------------------------------------------------------------------------------------------



void TSNE_FMM_SYM_MORTON_NN_Kernel(double& total, NodeFMM_MORTON_2D& sinkNode, NodeFMM_MORTON_2D& sourceNode)
{
    glm::vec2 R = sinkNode.centreOfMass - sourceNode.centreOfMass;
    float sq_r = R.x * R.x + R.y * R.y;
    float rS = 1.0f + sq_r;

    float D1 = 1.0f / (rS * rS);
    float D2 = -4.0f / (rS * rS * rS);
    float D3 = 24.0f / (rS * rS * rS * rS);

    float MA0 = sinkNode.M0;
    float MB0 = sourceNode.M0;
    Fastor::Tensor<float, 2, 2> MB2 = sourceNode.M2;
    Fastor::Tensor<float, 2, 2> MB2Tilde = (1.0f / MB0) * MB2;

    // calculate the C^m
    float MB2TildeSum1 = MB2Tilde(0, 0) + MB2Tilde(1, 1);
    float MB2TildeSum2 = (R.x * R.x * MB2Tilde(0, 0)) + (R.x * R.y * MB2Tilde(0, 1)) + (R.y * R.x * MB2Tilde(1, 0)) +
(R.y * R.y * MB2Tilde(1, 1)); float MB2TildeSum3i0 = R.x * MB2Tilde(0, 0) + R.y * MB2Tilde(0, 1); float MB2TildeSum3i1 =
R.x * MB2Tilde(1, 0) + R.y * MB2Tilde(1, 1);

    Fastor::Tensor<float, 2> C1 = MA0 * Fastor::Tensor<float, 2>
    {
        MB0 * (R.x * (D1 + 0.5f * (MB2TildeSum1)*D2 + 0.5f * (MB2TildeSum2)*D3) + (MB2TildeSum3i0)*D2),
        MB0 * (R.y * (D1 + 0.5f * (MB2TildeSum1)*D2 + 0.5f * (MB2TildeSum2)*D3) + (MB2TildeSum3i1)*D2)
    };

    Fastor::Tensor<float, 2, 2> C2 = MA0 * Fastor::Tensor<float, 2, 2>
    {
        {
            MB0 * (D1 + R.x * R.x * D2),
            MB0 * (R.x * R.y * D2)
        },
        {
            MB0 * (R.y * R.x * D2),
            MB0 * (D1 + R.y * R.y * D2)
        }
    };

    Fastor::Tensor<float, 2, 2, 2> C3 = MA0 * Fastor::Tensor<float, 2, 2, 2>
    {
        {
            {
                MB0 * ((R.x + R.x + R.x) * D2 + R.x * R.x * R.x * D3), // i = 0, j = 0, k = 0
                MB0 * ((R.y) * D2 + R.x * R.x * R.y * D3)  // i = 0, j = 0, k = 1
            },
            {
                MB0 * ((R.y) * D2 + R.x * R.y * R.x * D3), // i = 0, j = 1, k = 0
                MB0 * ((R.x) * D2 + R.x * R.y * R.y * D3)  // i = 0, j = 1, k = 1
            }
        },
        {
            {
                MB0 * ((R.y) * D2 + R.y * R.x * R.x * D3), // i = 1, j = 0, k = 0
                MB0 * ((R.x) * D2 + R.y * R.x * R.y * D3)  // i = 1, j = 0, k = 1
            },
            {
                MB0 * ((R.x) * D2 + R.y * R.y * R.x * D3), // i = 1, j = 1, k = 0
                MB0 * ((R.y + R.y + R.y) * D2 + R.y * R.y * R.y * D3)  // i = 1, j = 1, k = 1
            }
        }
    };


    sinkNode.C1 += C1;
    sinkNode.C2 += C2;
    sinkNode.C3 += C3;

    // second interaction

    R = -R;

    MA0 = sourceNode.M0;
    MB0 = sinkNode.M0;
    MB2 = sinkNode.M2;
    MB2Tilde = (1.0f / MB0) * MB2;

    // calculate the C^m
    MB2TildeSum1 = MB2Tilde(0, 0) + MB2Tilde(1, 1);
    MB2TildeSum2 = (R.x * R.x * MB2Tilde(0, 0)) + (R.x * R.y * MB2Tilde(0, 1)) + (R.y * R.x * MB2Tilde(1, 0)) + (R.y *
R.y * MB2Tilde(1, 1)); MB2TildeSum3i0 = R.x * MB2Tilde(0, 0) + R.y * MB2Tilde(0, 1); MB2TildeSum3i1 = R.x * MB2Tilde(1,
0) + R.y * MB2Tilde(1, 1);

    C1 = MA0 * Fastor::Tensor<float, 2>
    {
        MB0 * (R.x* (D1 + 0.5f * (MB2TildeSum1)*D2 + 0.5f * (MB2TildeSum2)*D3) + (MB2TildeSum3i0)*D2),
        MB0 * (R.y* (D1 + 0.5f * (MB2TildeSum1)*D2 + 0.5f * (MB2TildeSum2)*D3) + (MB2TildeSum3i1)*D2)
    };

    sourceNode.C1 += C1;
    sourceNode.C2 += C2;
    sourceNode.C3 += -C3;

    // add totals for both interactions

    total += static_cast<double>(2.0f * (sinkNode.M0 * sourceNode.M0) / rS);
}


void TSNE_FMM_SYM_MORTON_PN_Kernel(double& total, TsnePoint2D& sinkPoint, NodeFMM_MORTON_2D& sourceNode)
{
    glm::vec2 R = sinkPoint.position - sourceNode.centreOfMass;
    float sq_r = R.x * R.x + R.y * R.y;
    float rS = 1.0f + sq_r;

    float D1 = 1.0f / (rS * rS);
    float D2 = -4.0f / (rS * rS * rS);
    float D3 = 24.0f / (rS * rS * rS * rS);

    float MA0 = 1.0f;
    float MB0 = sourceNode.M0;
    Fastor::Tensor<float, 2, 2> MB2 = sourceNode.M2;
    Fastor::Tensor<float, 2, 2> MB2Tilde = (1.0f / MB0) * MB2;


    float MB2TildeSum1 = MB2Tilde(0, 0) + MB2Tilde(1, 1);
    float MB2TildeSum2 = (R.x * R.x * MB2Tilde(0, 0)) + (R.x * R.y * MB2Tilde(0, 1)) + (R.y * R.x * MB2Tilde(1, 0)) +
(R.y * R.y * MB2Tilde(1, 1)); float MB2TildeSum3i0 = R.x * MB2Tilde(0, 0) + R.y * MB2Tilde(0, 1); float MB2TildeSum3i1 =
R.x * MB2Tilde(1, 0) + R.y * MB2Tilde(1, 1);

    Fastor::Tensor<float, 2> C1 = MA0 * Fastor::Tensor<float, 2>
    {
        MB0 * (R.x * (D1 + 0.5f * (MB2TildeSum1)*D2 + 0.5f * (MB2TildeSum2)*D3) + (MB2TildeSum3i0)*D2),
        MB0 * (R.y * (D1 + 0.5f * (MB2TildeSum1)*D2 + 0.5f * (MB2TildeSum2)*D3) + (MB2TildeSum3i1)*D2)
    };

    sinkPoint.derivative += glm::vec2(C1(0), C1(1));

    // second interaction

    R = -R;

    MA0 = sourceNode.M0;
    MB0 = 1.0f;

    C1 = MA0 * Fastor::Tensor<float, 2>
    {
        (R.x * D1),
        (R.y * D1)
    };

    Fastor::Tensor<float, 2, 2> C2 = MA0 * Fastor::Tensor<float, 2, 2>
    {
        {
            (D1 + R.x * R.x * D2),
            (R.x * R.y * D2)
        },
        {
            (R.y * R.x * D2),
            (D1 + R.y * R.y * D2)
        }
    };

    Fastor::Tensor<float, 2, 2, 2> C3 = MA0 * Fastor::Tensor<float, 2, 2, 2>
    {
        {
            {
                ((R.x + R.x + R.x) * D2 + R.x * R.x * R.x * D3), // i = 0, j = 0, k = 0
                ((R.y) * D2 + R.x * R.x * R.y * D3)  // i = 0, j = 0, k = 1
            },
            {
                ((R.y) * D2 + R.x * R.y * R.x * D3), // i = 0, j = 1, k = 0
                ((R.x) * D2 + R.x * R.y * R.y * D3)  // i = 0, j = 1, k = 1
            }
        },
        {
            {
                ((R.y) * D2 + R.y * R.x * R.x * D3), // i = 1, j = 0, k = 0
                ((R.x) * D2 + R.y * R.x * R.y * D3)  // i = 1, j = 0, k = 1
            },
            {
                ((R.x) * D2 + R.y * R.y * R.x * D3), // i = 1, j = 1, k = 0
                ((R.y + R.y + R.y) * D2 + R.y * R.y * R.y * D3)  // i = 1, j = 1, k = 1
            }
        }
    };

    sourceNode.C1 += C1;
    sourceNode.C2 += C2;
    sourceNode.C3 += C3;

    // add totals for both interactions

    total += static_cast<double>((1.0f + sourceNode.M0) / rS);
}


void TSNE_FMM_SYM_MORTON_PP_Kernel(double& total, TsnePoint2D& sinkPoint, TsnePoint2D& sourcePoint)
{
    glm::vec2 R = sinkPoint.position - sourcePoint.position;
    float sq_dist = R.x * R.x + R.y * R.y;

    float forceDecay = 1.0f / (1.0f + sq_dist);

    sinkPoint.derivative += forceDecay * forceDecay * R;

    // second interaction

    R = -R;

    sourcePoint.derivative += forceDecay * forceDecay * R;

    // add totals for both interactions

    total += static_cast<double>(2.0f * forceDecay);
}
*/
