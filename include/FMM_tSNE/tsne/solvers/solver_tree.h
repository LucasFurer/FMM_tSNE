#pragma once

#include <FMM_tSNE/tsne/solvers/node.h>
#include <FMM_tSNE/tsne/solvers/solver.h>
#include <functional>
#include <glm/glm.hpp>
#include <utility>
#include <vector>

// #include "../../common.h"
// #include "nBodySolver.h"
// #include "../../trees/cpu/quadtree.h"
// #include "../../particles/embeddedPoint.h"
// #include "../../particles/Particle2D.h"
// #include "../../policies/BH_policy.h"
// #include "../../policies/rBH_policy.h"
// #include "../../policies/MP_policy.h"
// #include "../../policies/rMP_policy.h"

template <typename T, typename Policy>
class NBodySolverTree : public NBodySolver<T>
{
  public:
    QuadTree<T, Policy> root;

    NBodySolverTree() = default;

    NBodySolverTree(int initMaxChildren, double initTheta)
    {
        this->maxChildren = initMaxChildren;
        this->theta = initTheta;
    }

    void solveNbody(double& total, std::vector<T>& points) override
    {
        total = 0.0;

        Policy::traverse_tree(total, points, &root, this->theta);
    }

    void updateTree(std::vector<T>& points, glm::dvec2 minPos, glm::dvec2 maxPos) override
    {
        root = std::move(QuadTree<T, Policy>(this->maxChildren, points));
    }

    std::vector<VertexPos2Col3> getNodesBufferData(int nodeLevelToShow) override
    {
        std::vector<VertexPos2Col3> result;
        root.getNodesBufferData(result, 0, nodeLevelToShow);
        return result;
    }
};

// void TSNE_MP_PN_Kernel(double& total, TsnePoint2D& sinkPoint, QuadTree<TsnePoint2D, Policy_MP>* sourceNode)
//{
//     glm::dvec2 R = sinkPoint.position - sourceNode->summary.COM;
//     double sq_r = R.x * R.x + R.y * R.y;
//     double rS = 1.0 + sq_r;
//
//     double D1 = 1.0 / (rS * rS);
//     double D2 = -4.0 / (rS * rS * rS);
//     double D3 = 24.0 / (rS * rS * rS * rS);
//     total += sourceNode->summary.M0 / rS;
//
//     double MB0 = sourceNode->summary.M0;
//     Fastor::Tensor<double, 2, 2> MB2 = sourceNode->summary.M2;
//     Fastor::Tensor<double, 2, 2> MB2Tilde = (1.0 / MB0) * MB2;
//
//     double MB2TildeSum1 = MB2Tilde(0, 0) + MB2Tilde(1, 1);
//     double MB2TildeSum2 = (R.x * R.x * MB2Tilde(0, 0)) + (R.x * R.y * MB2Tilde(0, 1)) + (R.y * R.x * MB2Tilde(1, 0))
//     + (R.y * R.y * MB2Tilde(1, 1)); double MB2TildeSum3i0 = R.x * MB2Tilde(0, 0) + R.y * MB2Tilde(0, 1); double
//     MB2TildeSum3i1 = R.x * MB2Tilde(1, 0) + R.y * MB2Tilde(1, 1);
//
//     Fastor::Tensor<double, 2> C1 =
//     {
//         MB0 * (R.x * (D1 + 0.5 * (MB2TildeSum1)*D2 + 0.5 * (MB2TildeSum2)*D3) + (MB2TildeSum3i0)*D2),
//         MB0 * (R.y * (D1 + 0.5 * (MB2TildeSum1)*D2 + 0.5 * (MB2TildeSum2)*D3) + (MB2TildeSum3i1)*D2)
//     };
//
//     sinkPoint.derivative += glm::dvec2(C1(0), C1(1));
// }
//
// void TSNE_MP_PP_Kernel(double& total, TsnePoint2D& sinkPoint, TsnePoint2D& sourcePoint)
//{
//     glm::dvec2 diff = sinkPoint.position - sourcePoint.position;
//     double sq_dist = diff.x * diff.x + diff.y * diff.y;
//
//     double forceDecay = 1.0 / (1.0 + sq_dist);
//     total += forceDecay;
//
//     sinkPoint.derivative += forceDecay * forceDecay * diff;
// }
