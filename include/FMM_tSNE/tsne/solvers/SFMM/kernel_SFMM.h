#pragma once

#include <FMM_tSNE/tsne/points/tsne_point_2D.h>
#include <FMM_tSNE/tsne/solvers/SFMM/node_SFMM.h>
#include <glm/glm.hpp>
#include <glm/gtx/component_wise.hpp>
#include <glm/gtx/string_cast.hpp>

inline void TSNE_SFMM_NN(double& total, NodeSFMM& node_A, NodeSFMM& node_B)
{
    const glm::dvec2 R_BA = node_A.COM - node_B.COM;
    const double R_xx = R_BA.x * R_BA.x;
    const double R_xy = R_BA.x * R_BA.y;
    const double R_yy = R_BA.y * R_BA.y;
    const double r_sq = R_xx + R_yy;

    const double rS = 1.0 + r_sq;
    const double inv_rS = 1.0 / rS;
    const double inv_rS2 = inv_rS * inv_rS;
    const double inv_rS3 = inv_rS2 * inv_rS;
    const double inv_rS4 = inv_rS3 * inv_rS;

    const double D1 = inv_rS2;
    const double D2 = -4.0 * inv_rS3;
    const double D3 = 24.0 * inv_rS4;

    const double MA0 = node_A.M0;
    const double MB0 = node_B.M0;

    const double one_over_MB0 = 1.0 / MB0;
    const double MB2Tilde_xx = one_over_MB0 * node_B.M2xx;
    const double MB2Tilde_xy = one_over_MB0 * node_B.M2xy;
    const double MB2Tilde_yy = one_over_MB0 * node_B.M2yy;

    // calculate the C^m
    const double MB2TildeSum1 = MB2Tilde_xx + MB2Tilde_yy;
    const double MB2TildeSum2 =
        (R_xx * MB2Tilde_xx) + (R_xy * MB2Tilde_xy) + (R_xy * MB2Tilde_xy) + (R_yy * MB2Tilde_yy);
    const double MB2TildeSum3i0 = R_BA.x * MB2Tilde_xx + R_BA.y * MB2Tilde_xy;
    const double MB2TildeSum3i1 = R_BA.x * MB2Tilde_xy + R_BA.y * MB2Tilde_yy;

    const MF field = {
        MB0 * (R_BA.x * (D1 + (0.5 * MB2TildeSum1 * D2) + (0.5 * MB2TildeSum2 * D3)) + (MB2TildeSum3i0 * D2)),
        MB0 * (R_BA.y * (D1 + (0.5 * MB2TildeSum1 * D2) + (0.5 * MB2TildeSum2 * D3)) + (MB2TildeSum3i1 * D2)),
        MA0 * MB0 * (D1 + (R_xx * D2)),
        MA0 * MB0 * (R_xy * D2),
        MA0 * MB0 * (D1 + (R_yy * D2)),
        MA0 * MB0 * ((R_BA.x + R_BA.x + R_BA.x) * D2 + (R_xx * R_BA.x * D3)),
        MA0 * MB0 * ((R_BA.y * D2) + (R_xx * R_BA.y * D3)),
        MA0 * MB0 * ((R_BA.x * D2) + (R_xy * R_BA.y * D3)),
        MA0 * MB0 * ((R_BA.y + R_BA.y + R_BA.y) * D2 + (R_yy * R_BA.y * D3)),
    };
    node_A.field += field;

    // second interaction ---------------------------------------------------------------------------------------

    const glm::dvec2 R_AB = -R_BA;

    const double one_over_MA0 = 1.0 / MA0;
    const double MA2Tilde_xx = one_over_MA0 * node_A.M2xx;
    const double MA2Tilde_xy = one_over_MA0 * node_A.M2xy;
    const double MA2Tilde_yy = one_over_MA0 * node_A.M2yy;

    // calculate the C^m
    const double MA2TildeSum1 = MA2Tilde_xx + MA2Tilde_yy;
    const double MA2TildeSum2 =
        (R_xx * MA2Tilde_xx) + (R_xy * MA2Tilde_xy) + (R_xy * MA2Tilde_xy) + (R_yy * MA2Tilde_yy);
    const double MA2TildeSum3i0 = R_AB.x * MA2Tilde_xx + R_AB.y * MA2Tilde_xy;
    const double MA2TildeSum3i1 = R_AB.x * MA2Tilde_xy + R_AB.y * MA2Tilde_yy;

    node_B.field +=
        {MA0 * (R_AB.x * (D1 + (0.5 * MA2TildeSum1 * D2) + (0.5 * MA2TildeSum2 * D3)) + (MA2TildeSum3i0 * D2)),
         MA0 * (R_AB.y * (D1 + (0.5 * MA2TildeSum1 * D2) + (0.5 * MA2TildeSum2 * D3)) + (MA2TildeSum3i1 * D2)),
         field.C2xx,
         field.C2xy,
         field.C2yy,
         -field.C3xxx,
         -field.C3xxy,
         -field.C3xyy,
         -field.C3yyy};

    // add totals for both interactions
    total += 2.0 * (MA0 * MB0) / rS;
}

inline void TSNE_SFMM_PN(double& total, TsnePoint2D& point_A, NodeSFMM& node_B)
{
    const glm::dvec2 R_BA = point_A.position - node_B.COM;
    const double R_xx = R_BA.x * R_BA.x;
    const double R_xy = R_BA.x * R_BA.y;
    const double R_yy = R_BA.y * R_BA.y;
    const double r_sq = R_xx + R_yy;

    const double rS = 1.0 + r_sq;
    const double inv_rS = 1.0 / rS;
    const double inv_rS2 = inv_rS * inv_rS;
    const double inv_rS3 = inv_rS2 * inv_rS;
    const double inv_rS4 = inv_rS3 * inv_rS;

    const double D1 = inv_rS2;
    const double D2 = -4.0 * inv_rS3;
    const double D3 = 24.0 * inv_rS4;

    const double MA0 = 1.0;
    const double MB0 = node_B.M0;

    const double one_over_MB0 = 1.0 / MB0;
    const double MB2Tilde_xx = one_over_MB0 * node_B.M2xx;
    const double MB2Tilde_xy = one_over_MB0 * node_B.M2xy;
    const double MB2Tilde_yy = one_over_MB0 * node_B.M2yy;

    const double MB2TildeSum1 = MB2Tilde_xx + MB2Tilde_yy;
    const double MB2TildeSum2 =
        (R_xx * MB2Tilde_xx) + (R_xy * MB2Tilde_xy) + (R_xy * MB2Tilde_xy) + (R_yy * MB2Tilde_yy);
    const double MB2TildeSum3i0 = R_BA.x * MB2Tilde_xx + R_BA.y * MB2Tilde_xy;
    const double MB2TildeSum3i1 = R_BA.x * MB2Tilde_xy + R_BA.y * MB2Tilde_yy;

    point_A.derivative +=
        glm::dvec2{MB0 * (R_BA.x * (D1 + 0.5 * (MB2TildeSum1) *D2 + 0.5 * (MB2TildeSum2) *D3) + (MB2TildeSum3i0) *D2),
                   MB0 * (R_BA.y * (D1 + 0.5 * (MB2TildeSum1) *D2 + 0.5 * (MB2TildeSum2) *D3) + (MB2TildeSum3i1) *D2)};

    // second interaction ---------------------------------------------------------------------------------------

    const glm::dvec2 R_AB = -R_BA;

    node_B.field += {R_AB.x * D1,
                     R_AB.y * D1,
                     MB0 * (D1 + (R_xx * D2)),
                     MB0 * (R_xy * D2),
                     MB0 * (D1 + (R_yy * D2)),
                     MB0 * ((R_AB.x + R_AB.x + R_AB.x) * D2 + (R_xx * R_AB.x * D3)),
                     MB0 * ((R_AB.y * D2) + (R_xx * R_AB.y * D3)),
                     MB0 * ((R_AB.x * D2) + (R_xy * R_AB.y * D3)),
                     MB0 * ((R_AB.y + R_AB.y + R_AB.y) * D2 + (R_yy * R_AB.y * D3))};

    // add totals for both interactions
    total += 2.0 * MB0 / rS;
}

inline void TSNE_SFMM_PP(double& total, TsnePoint2D& point_A, TsnePoint2D& point_B)
{
    const glm::dvec2 R_BA = point_A.position - point_B.position;
    const double r_sq = R_BA.x * R_BA.x + R_BA.y * R_BA.y;

    const double rS = 1.0 + r_sq;

    const double inv_rS = 1.0 / rS;

    point_A.derivative += inv_rS * inv_rS * R_BA;

    // second interaction ---------------------------------------------------------------------------------------

    const glm::dvec2 R_AB = -R_BA;

    point_B.derivative += inv_rS * inv_rS * R_AB;

    // add totals for both interactions
    total += 2.0 * inv_rS;
}
