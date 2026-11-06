#include <FMM_tSNE/tsne/points/tsne_point_2D.h>
#include <FMM_tSNE/tsne/solvers/FMM/node_FMM.h>
#include <glm/glm.hpp>
#include <glm/gtx/component_wise.hpp>
#include <glm/gtx/string_cast.hpp>

inline void TSNE_FMM_NN(double& total, NodeFMM& sinkNode, MF& sinkField, NodeFMM& sourceNode)
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
    double MB2_xx = sourceNode.M2xx;
    double MB2_xy = sourceNode.M2xy;
    double MB2_yy = sourceNode.M2yy;
    double MB2Tilde_xx = (1.0 / MB0) * MB2_xx;
    double MB2Tilde_xy = (1.0 / MB0) * MB2_xy;
    double MB2Tilde_yy = (1.0 / MB0) * MB2_yy;

    // calculate the C^m
    double MB2TildeSum1 = MB2Tilde_xx + MB2Tilde_yy;
    double MB2TildeSum2 =
        (R.x * R.x * MB2Tilde_xx) + (R.x * R.y * MB2Tilde_xy) + (R.y * R.x * MB2Tilde_xy) + (R.y * R.y * MB2Tilde_yy);
    double MB2TildeSum3i0 = R.x * MB2Tilde_xx + R.y * MB2Tilde_xy;
    double MB2TildeSum3i1 = R.x * MB2Tilde_xy + R.y * MB2Tilde_yy;

    sinkField += {MB0 * (R.x * (D1 + 0.5 * (MB2TildeSum1) *D2 + 0.5 * (MB2TildeSum2) *D3) + (MB2TildeSum3i0) *D2),
                  MB0 * (R.y * (D1 + 0.5 * (MB2TildeSum1) *D2 + 0.5 * (MB2TildeSum2) *D3) + (MB2TildeSum3i1) *D2),
                  MA0 * MB0 * (D1 + R.x * R.x * D2),
                  MA0 * MB0 * (R.x * R.y * D2),
                  MA0 * MB0 * (D1 + R.y * R.y * D2),
                  MA0 * MB0 * ((R.x + R.x + R.x) * D2 + R.x * R.x * R.x * D3),
                  MA0 * MB0 * ((R.y) * D2 + R.x * R.x * R.y * D3),
                  MA0 * MB0 * ((R.x) * D2 + R.x * R.y * R.y * D3),
                  MA0 * MB0 * ((R.y + R.y + R.y) * D2 + R.y * R.y * R.y * D3)};

    // add totals for both interactions
    total += (sinkNode.M0 * sourceNode.M0) / rS;
}

inline void TSNE_FMM_PN(double& total, TsnePoint2D& sinkPoint, NodeFMM& sourceNode)
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
    double MB2_xx = sourceNode.M2xx;
    double MB2_xy = sourceNode.M2xy;
    double MB2_yy = sourceNode.M2yy;
    double MB2Tilde_xx = (1.0 / MB0) * MB2_xx;
    double MB2Tilde_xy = (1.0 / MB0) * MB2_xy;
    double MB2Tilde_yy = (1.0 / MB0) * MB2_yy;

    double MB2TildeSum1 = MB2Tilde_xx + MB2Tilde_yy;
    double MB2TildeSum2 =
        (R.x * R.x * MB2Tilde_xx) + (R.x * R.y * MB2Tilde_xy) + (R.y * R.x * MB2Tilde_xy) + (R.y * R.y * MB2Tilde_yy);
    double MB2TildeSum3i0 = R.x * MB2Tilde_xx + R.y * MB2Tilde_xy;
    double MB2TildeSum3i1 = R.x * MB2Tilde_xy + R.y * MB2Tilde_yy;

    sinkPoint.derivative +=
        glm::dvec2{MB0 * (R.x * (D1 + 0.5 * (MB2TildeSum1) *D2 + 0.5 * (MB2TildeSum2) *D3) + (MB2TildeSum3i0) *D2),
                   MB0 * (R.y * (D1 + 0.5 * (MB2TildeSum1) *D2 + 0.5 * (MB2TildeSum2) *D3) + (MB2TildeSum3i1) *D2)};

    total += sourceNode.M0 / rS;
}

inline void TSNE_FMM_NP(double& total, NodeFMM& sinkNode, MF& sinkField, TsnePoint2D& sourcePoint)
{
    glm::dvec2 R = sinkNode.COM - sourcePoint.position;
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
    double MB0 = 1.0;

    sinkField += {(R.x * D1),
                  (R.y * D1),
                  MA0 * (D1 + R.x * R.x * D2),
                  MA0 * (R.x * R.y * D2),
                  MA0 * (D1 + R.y * R.y * D2),
                  MA0 * ((R.x + R.x + R.x) * D2 + R.x * R.x * R.x * D3),
                  MA0 * ((R.y) * D2 + R.x * R.x * R.y * D3),
                  MA0 * ((R.x) * D2 + R.x * R.y * R.y * D3),
                  MA0 * ((R.y + R.y + R.y) * D2 + R.y * R.y * R.y * D3)};

    total += sinkNode.M0 / rS;
}

inline void TSNE_FMM_PP(double& total, TsnePoint2D& sinkPoint, TsnePoint2D& sourcePoint)
{
    glm::dvec2 R = sinkPoint.position - sourcePoint.position;
    double sq_dist = R.x * R.x + R.y * R.y;

    double forceDecay = 1.0 / (1.0 + sq_dist);

    sinkPoint.derivative += forceDecay * forceDecay * R;

    total += forceDecay;
}
