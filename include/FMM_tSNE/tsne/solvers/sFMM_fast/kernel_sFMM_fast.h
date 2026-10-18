#include <FMM_tSNE/timer/timer.h>
#include <FMM_tSNE/tsne/points/tsne_point_2D.h>
#include <FMM_tSNE/tsne/solvers/sFMM/node_sFMM.h>
#include <FMM_tSNE/tsne/solvers/sFMM_fast/node_sFMM_fast.h>
#include <FMM_tSNE/tsne/solvers/solver.h>
#include <Fastor/Fastor.h>
#include <boost/sort/sort.hpp>
#include <glm/glm.hpp>
#include <glm/gtx/component_wise.hpp>
#include <glm/gtx/string_cast.hpp>

inline void TSNE_sFMMfast_NN(double& total, NodeSFMMFast& sinkNode, NodeSFMMFast& sourceNode)
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

inline void TSNE_sFMMfast_PN(double& total, TsnePoint2D& sinkPoint, NodeSFMMFast& sourceNode)
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

inline void TSNE_sFMMfast_PP(double& total, TsnePoint2D& sinkPoint, TsnePoint2D& sourcePoint)
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
