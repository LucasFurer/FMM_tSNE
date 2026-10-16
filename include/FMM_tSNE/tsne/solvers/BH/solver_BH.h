#pragma once

#include <Fastor/Fastor.h>
#include <algorithm>
#include <glm/glm.hpp>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

struct Policy_BH
{
    struct Node_Summary
    {
        glm::dvec2 COM{0.0};

        double M0{0.0};
    };

    template <typename T, typename QuadTreeType>
    static inline std::pair<double, glm::vec2> compute_node(std::vector<T>& allParticles,
                                                            int maxChildren,
                                                            Node_Summary& summary,
                                                            const std::vector<QuadTreeType*>& children)
    {
        for (auto* child : children)
        {
            std::pair<double, glm::dvec2> childMassPosition = child->createTree(allParticles, maxChildren);
            summary.M0 += childMassPosition.first;
            summary.COM += childMassPosition.first * childMassPosition.second;
        }

        if (summary.M0 != 0.0) summary.COM /= summary.M0;

        return std::make_pair(summary.M0, summary.COM);
    }

    template <typename T>
    static inline std::pair<double, glm::vec2> compute_leaf(std::vector<T>& allParticles,
                                                            int maxChildren,
                                                            Node_Summary& summary,
                                                            const std::vector<int>& occupants)
    {
        for (int i = 0; i < occupants.size(); i++)
        {
            summary.M0 += 1.0;
            summary.COM += allParticles[occupants[i]].position;
        }

        if (summary.M0 != 0.0) summary.COM /= summary.M0;

        return std::make_pair(summary.M0, summary.COM);
    }

    // interactions----------------------------------------------------------

    template <typename T, typename QuadTreeType>
    static inline void PN_interaction(double& total, T& sinkPoint, QuadTreeType* sourceNode)
    {
        glm::dvec2 diff = sinkPoint.position - sourceNode->summary.COM;
        double sq_dist = diff.x * diff.x + diff.y * diff.y;

        double forceDecay = 1.0 / (1.0 + sq_dist);
        total += sourceNode->summary.M0 * forceDecay;

        sinkPoint.derivative += sourceNode->summary.M0 * forceDecay * forceDecay * diff;
    }

    template <typename T>
    static inline void PP_interaction(double& total, T& sinkPoint, T& sourcePoint)
    {
        glm::dvec2 diff = sinkPoint.position - sourcePoint.position;
        double sq_dist = diff.x * diff.x + diff.y * diff.y;

        double forceDecay = 1.0 / (1.0 + sq_dist);
        total += forceDecay;

        sinkPoint.derivative += forceDecay * forceDecay * diff;
    }

    // tree traversal----------------------------------------------------------

    template <typename T, typename QuadTreeType>
    static inline void traverse_tree(double& total, std::vector<T>& points, QuadTreeType* root, double theta)
    {
        for (int i = 0; i < points.size(); i++)
        {
            Policy_BH::traverseBH(total, points, points[i], root, theta);
        }
    }

    template <typename T, typename QuadTreeType>
    static inline void
    traverseBH(double& total, std::vector<T>& allParticles, T& point, QuadTreeType* node, double theta)
    {
        glm::dvec2 diff = point.position - node->summary.COM;

        if (node->BBlength / glm::length(diff) <
            theta) // && (glm::any(glm::lessThan(point.position, node->lowestCorner)) ||
                   // glm::any(glm::greaterThan(point.position, node->highestCorner))))
        {

            Policy_BH::PN_interaction(total, point, node);
        }
        else if (node->children.size() <= 1)
        {
            for (int i : node->occupants)
            {
                if (&allParticles[i] != &point) // self intersection test
                {

                    Policy_BH::PP_interaction(total, point, allParticles[i]);
                }
            }
        }
        else
        {
            for (QuadTreeType* child : node->children)
            {

                Policy_BH::traverseBH(total, allParticles, point, child, theta);
            }
        }
    }
};
