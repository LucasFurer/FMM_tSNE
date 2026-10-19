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

    sinkNode.C1x += MB0 * (R.x * (D1 + 0.5 * (MB2TildeSum1) *D2 + 0.5 * (MB2TildeSum2) *D3) + (MB2TildeSum3i0) *D2);
    sinkNode.C1y += MB0 * (R.y * (D1 + 0.5 * (MB2TildeSum1) *D2 + 0.5 * (MB2TildeSum2) *D3) + (MB2TildeSum3i1) *D2);

    const double C2xx = MA0 * MB0 * (D1 + R.x * R.x * D2);
    const double C2xy = MA0 * MB0 * (R.x * R.y * D2);
    const double C2yy = MA0 * MB0 * (D1 + R.y * R.y * D2);

    sinkNode.C2xx += C2xx;
    sinkNode.C2xy += C2xy;
    sinkNode.C2yy += C2yy;

    const double C3xxx = MA0 * MB0 * ((R.x + R.x + R.x) * D2 + R.x * R.x * R.x * D3);
    const double C3xxy = MA0 * MB0 * ((R.y) * D2 + R.x * R.x * R.y * D3);
    const double C3xyy = MA0 * MB0 * ((R.x) * D2 + R.x * R.y * R.y * D3);
    const double C3yyy = MA0 * MB0 * ((R.y + R.y + R.y) * D2 + R.y * R.y * R.y * D3);

    sinkNode.C3xxx += C3xxx;
    sinkNode.C3xxy += C3xxy;
    sinkNode.C3xyy += C3xyy;
    sinkNode.C3yyy += C3yyy;

    // second interaction ---------------------------------------------------------------------------------------

    R = -R;

    MA0 = sourceNode.M0;
    MB0 = sinkNode.M0;
    MB2_xx = sinkNode.M2xx;
    MB2_xy = sinkNode.M2xy;
    MB2_yy = sinkNode.M2yy;
    MB2Tilde_xx = (1.0 / MB0) * MB2_xx;
    MB2Tilde_xy = (1.0 / MB0) * MB2_xy;
    MB2Tilde_yy = (1.0 / MB0) * MB2_yy;

    // calculate the C^m
    MB2TildeSum1 = MB2Tilde_xx + MB2Tilde_yy;
    MB2TildeSum2 =
        (R.x * R.x * MB2Tilde_xx) + (R.x * R.y * MB2Tilde_xy) + (R.y * R.x * MB2Tilde_xy) + (R.y * R.y * MB2Tilde_yy);
    MB2TildeSum3i0 = R.x * MB2Tilde_xx + R.y * MB2Tilde_xy;
    MB2TildeSum3i1 = R.x * MB2Tilde_xy + R.y * MB2Tilde_yy;

    sourceNode.C1x += MB0 * (R.x * (D1 + 0.5 * (MB2TildeSum1) *D2 + 0.5 * (MB2TildeSum2) *D3) + (MB2TildeSum3i0) *D2);
    sourceNode.C1y += MB0 * (R.y * (D1 + 0.5 * (MB2TildeSum1) *D2 + 0.5 * (MB2TildeSum2) *D3) + (MB2TildeSum3i1) *D2);

    sourceNode.C2xx += C2xx;
    sourceNode.C2xy += C2xy;
    sourceNode.C2yy += C2yy;

    sourceNode.C3xxx -= C3xxx;
    sourceNode.C3xxy -= C3xxy;
    sourceNode.C3xyy -= C3xyy;
    sourceNode.C3yyy -= C3yyy;

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

    sinkPoint.derivative.x +=
        MB0 * (R.x * (D1 + 0.5 * (MB2TildeSum1) *D2 + 0.5 * (MB2TildeSum2) *D3) + (MB2TildeSum3i0) *D2);
    sinkPoint.derivative.y +=
        MB0 * (R.y * (D1 + 0.5 * (MB2TildeSum1) *D2 + 0.5 * (MB2TildeSum2) *D3) + (MB2TildeSum3i1) *D2);

    // second interaction ---------------------------------------------------------------------------------------

    R = -R;

    MA0 = sourceNode.M0;
    MB0 = 1.0;

    sourceNode.C1x += (R.x * D1);
    sourceNode.C1y += (R.y * D1);

    const double C2xx = MA0 * (D1 + R.x * R.x * D2);
    const double C2xy = MA0 * (R.x * R.y * D2);
    const double C2yy = MA0 * (D1 + R.y * R.y * D2);

    sourceNode.C2xx += C2xx;
    sourceNode.C2xy += C2xy;
    sourceNode.C2yy += C2yy;

    const double C3xxx = MA0 * ((R.x + R.x + R.x) * D2 + R.x * R.x * R.x * D3);
    const double C3xxy = MA0 * ((R.y) * D2 + R.x * R.x * R.y * D3);
    const double C3xyy = MA0 * ((R.x) * D2 + R.x * R.y * R.y * D3);
    const double C3yyy = MA0 * ((R.y + R.y + R.y) * D2 + R.y * R.y * R.y * D3);

    sourceNode.C3xxx += C3xxx;
    sourceNode.C3xxy += C3xxy;
    sourceNode.C3xyy += C3xyy;
    sourceNode.C3yyy += C3yyy;

    // add totals for both interactions
    total += 2.0 * sourceNode.M0 / rS;
}

inline void TSNE_sFMMfast_PP(double& total, TsnePoint2D& sinkPoint, TsnePoint2D& sourcePoint)
{
    glm::dvec2 R = sinkPoint.position - sourcePoint.position;
    double sq_dist = R.x * R.x + R.y * R.y;

    double forceDecay = 1.0 / (1.0 + sq_dist);

    sinkPoint.derivative += forceDecay * forceDecay * R;

    // second interaction ---------------------------------------------------------------------------------------

    R = -R;

    sourcePoint.derivative += forceDecay * forceDecay * R;

    // add totals for both interactions
    total += 2.0 * forceDecay;
}
