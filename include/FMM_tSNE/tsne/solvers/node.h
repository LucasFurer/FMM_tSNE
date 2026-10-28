#pragma once

#include <FMM_tSNE/tsne/points/vertex_pos2_col3.h>
#include <Fastor/Fastor.h>
#include <algorithm>
#include <glm/glm.hpp>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

// #include "../../policies/BH_policy.h"
// #include "../../policies/MP_policy.h"
// #include "../../policies/rBH_policy.h"
// #include "../../policies/rMP_policy.h"

template <typename T, typename Policy>
class QuadTree
{
  public:
    typename Policy::Node_Summary summary;

    glm::dvec2 BBcentre{0.0};
    double BBlength{0.0};

    std::vector<int> occupants{};
    std::vector<QuadTree*> children{};

    QuadTree() = default;

    QuadTree(int maxChildren, std::vector<T>& allParticles)
    {
        glm::dvec2 setLowestCorner(std::numeric_limits<double>::infinity());
        glm::dvec2 setHighestCorner(-std::numeric_limits<double>::infinity());

        for (int i = 0; i < allParticles.size(); i++)
        {
            occupants.push_back(i);

            setLowestCorner = glm::min(setLowestCorner, allParticles[i].position);
            setHighestCorner = glm::max(setHighestCorner, allParticles[i].position);
        }

        BBlength = std::max(setHighestCorner.x - setLowestCorner.x, setHighestCorner.y - setLowestCorner.y);
        BBcentre = setLowestCorner + 0.5 * BBlength;

        auto childMassPosition = createTree(allParticles, maxChildren);
    }

    QuadTree(std::vector<int> initOccupants, glm::dvec2 initBBcentre, double initBBlength)
    {
        occupants = initOccupants;

        BBcentre = initBBcentre;
        BBlength = initBBlength;
    }

    // Destructor
    ~QuadTree()
    {
        for (QuadTree* quadTree : children)
        {
            delete quadTree;
        }
    }

    // Copy constructor
    QuadTree(const QuadTree& other)
        : summary(other.summary),
          BBcentre(other.BBcentre),
          BBlength(other.BBlength),
          occupants(other.occupants)
    {
        children.reserve(other.children.size());
        for (QuadTree* quadTree : other.children)
        {
            children.push_back(new QuadTree(*quadTree));
        }
    }

    // swap
    friend void swap(QuadTree& a, QuadTree& b) noexcept
    {
        std::swap(a.summary, b.summary);
        std::swap(a.BBcentre, b.BBcentre);
        std::swap(a.BBlength, b.BBlength);
        std::swap(a.occupants, b.occupants);
        std::swap(a.children, b.children);
    }

    // Copy assignment
    QuadTree& operator=(const QuadTree& other)
    {
        if (this != &other) // check if not assigning to itself
        {
            QuadTree temp(other);
            swap(*this, temp);
        }
        return *this;
    }

    // Move constructor
    QuadTree(QuadTree&& other) noexcept
        : summary(other.summary),
          BBcentre(other.BBcentre),
          BBlength(other.BBlength),
          occupants(std::move(other.occupants)),
          children(std::move(other.children))
    {
        other.summary = {}; // clean up other (but not the moved items)
        other.BBcentre = glm::dvec2(0.0);
        other.BBlength = 0.0;
    }

    // Move assignment
    QuadTree& operator=(QuadTree&& other) noexcept
    {
        if (this != &other) // check if not assigning to itself
        {
            for (QuadTree* quadTreeBH : children)
            {
                delete quadTreeBH;
            } // clean up this

            summary = other.summary; // copy other to this
            BBcentre = other.BBcentre;
            BBlength = other.BBlength;
            occupants = std::move(other.occupants); // move other to this
            children = std::move(other.children);

            other.summary = {}; // clean up other (but not the moved items)
            other.BBcentre = glm::dvec2(0.0);
            other.BBlength = 0.0;
        }
        return *this;
    }

    auto createTree(std::vector<T>& allParticles, int maxChildren)
    {
        if (occupants.size() > maxChildren)
        {
            std::vector<int> HH;
            std::vector<int> HL;
            std::vector<int> LH;
            std::vector<int> LL;

            for (int i = 0; i < occupants.size(); i++)
            {
                int index = occupants[i];

                if (allParticles[index].position.x >= BBcentre.x && allParticles[index].position.y >= BBcentre.y)
                {
                    HH.push_back(occupants[i]);
                }
                else if (allParticles[index].position.x >= BBcentre.x && allParticles[index].position.y < BBcentre.y)
                {
                    HL.push_back(occupants[i]);
                }
                else if (allParticles[index].position.x < BBcentre.x && allParticles[index].position.y >= BBcentre.y)
                {
                    LH.push_back(occupants[i]);
                }
                else if (allParticles[index].position.x < BBcentre.x && allParticles[index].position.y < BBcentre.y)
                {
                    LL.push_back(occupants[i]);
                }
                else
                {
                    std::cout << "something is wrong in octtree" << std::endl;
                }
            }

            float new_BBlength = BBlength * 0.5;
            if (HH.size() != 0)
            {
                children.push_back(
                    new QuadTree(HH, BBcentre + glm::dvec2(BBlength * 0.25, BBlength * 0.25), new_BBlength));
            }
            if (HL.size() != 0)
            {
                children.push_back(
                    new QuadTree(HL, BBcentre + glm::dvec2(BBlength * 0.25, -BBlength * 0.25), new_BBlength));
            }
            if (LH.size() != 0)
            {
                children.push_back(
                    new QuadTree(LH, BBcentre + glm::dvec2(-BBlength * 0.25, BBlength * 0.25), new_BBlength));
            }
            if (LL.size() != 0)
            {
                children.push_back(
                    new QuadTree(LL, BBcentre + glm::dvec2(-BBlength * 0.25, -BBlength * 0.25), new_BBlength));
            }

            return Policy::compute_node(allParticles, maxChildren, summary, children);
        }
        else
        {
            return Policy::compute_leaf(allParticles, maxChildren, summary, occupants);
        }
    }

    void getNodesBufferData(std::vector<VertexPos2Col3>& nodesBufferData, int level, int showLevel)
    {
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
            glm::dvec2 lowestCorner = BBcentre - 0.5 * BBlength;
            glm::dvec2 highestCorner = BBcentre + 0.5 * BBlength;

            nodesBufferData.push_back(VertexPos2Col3(glm::vec2(lowestCorner.x, lowestCorner.y), color));
            nodesBufferData.push_back(VertexPos2Col3(glm::vec2(highestCorner.x, lowestCorner.y), color));

            nodesBufferData.push_back(VertexPos2Col3(glm::vec2(lowestCorner.x, lowestCorner.y), color));
            nodesBufferData.push_back(VertexPos2Col3(glm::vec2(lowestCorner.x, highestCorner.y), color));

            nodesBufferData.push_back(VertexPos2Col3(glm::vec2(lowestCorner.x, highestCorner.y), color));
            nodesBufferData.push_back(VertexPos2Col3(glm::vec2(highestCorner.x, highestCorner.y), color));

            nodesBufferData.push_back(VertexPos2Col3(glm::vec2(highestCorner.x, lowestCorner.y), color));
            nodesBufferData.push_back(VertexPos2Col3(glm::vec2(highestCorner.x, highestCorner.y), color));
        }

        for (QuadTree* octTree : children)
        {
            octTree->getNodesBufferData(nodesBufferData, level + 1, showLevel);
        }
    }
};
