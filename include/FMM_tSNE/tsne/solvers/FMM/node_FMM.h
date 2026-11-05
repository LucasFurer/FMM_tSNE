#pragma once

#include "FMM_tSNE/tsne/points/tsne_point_2D.h"
#include "glm/ext/vector_double2.hpp"
#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>

struct NodeFMM;

struct MF
{
    double C1x;
    double C1y;
    double C2xx;
    double C2xy;
    double C2yy;
    double C3xxx;
    double C3xxy;
    double C3xyy;
    double C3yyy;

    MF()
        : C1x{},
          C1y{},
          C2xx{},
          C2xy{},
          C2yy{},
          C3xxx{},
          C3xxy{},
          C3xyy{},
          C3yyy{}
    {
    }

    MF(double in_C1x,
       double in_C1y,
       double in_C2xx,
       double in_C2xy,
       double in_C2yy,
       double in_C3xxx,
       double in_C3xxy,
       double in_C3xyy,
       double in_C3yyy)
        : C1x{in_C1x},
          C1y{in_C1y},
          C2xx{in_C2xx},
          C2xy{in_C2xy},
          C2yy{in_C2yy},
          C3xxx{in_C3xxx},
          C3xxy{in_C3xxy},
          C3xyy{in_C3xyy},
          C3yyy{in_C3yyy}
    {
    }

    MF& operator+=(const MF& o)
    {
        C1x += o.C1x;
        C1y += o.C1y;
        C2xx += o.C2xx;
        C2xy += o.C2xy;
        C2yy += o.C2yy;
        C3xxx += o.C3xxx;
        C3xxy += o.C3xxy;
        C3xyy += o.C3xyy;
        C3yyy += o.C3yyy;

        return *this;
    }

    void normalize(double norm_val)
    {
        // double norm_val = 1.0 / M0;
        // we dont normalize C1
        C2xx *= norm_val;
        C2xy *= norm_val;
        C2yy *= norm_val;
        C3xxx *= norm_val;
        C3xxy *= norm_val;
        C3xyy *= norm_val;
        C3yyy *= norm_val;
    }

    static void translate_from_to_node(NodeFMM& from, NodeFMM& to);

    static void translate_from_to_point(NodeFMM& from, TsnePoint2D& to);
};

struct NodeFMM
{
    glm::dvec2 COM{};

    double M0{};
    double M2xx{};
    double M2xy{};
    double M2yy{};

    double BB_length;

    unsigned int first_particle_index;  // first_particle_index + particle_index_amount is index of last particle
    unsigned int particle_index_amount; // amount of particles in node

    unsigned int first_child_index;   // first_child_index + child_index_amount is index of last child
    unsigned char child_index_amount; // amount of children (max is 4)

    MF field{};

    NodeFMM(double in_BB_length, unsigned int in_first_particle_index, unsigned int in_particle_index_amount)
        : BB_length{in_BB_length},
          first_particle_index{in_first_particle_index},
          particle_index_amount{in_particle_index_amount}
    {
    }
};

inline void MF::translate_from_to_node(NodeFMM& from, NodeFMM& to)
{
    glm::dvec2 oldZ = to.COM;
    glm::dvec2 newZ = from.COM;
    glm::dvec2 diff = oldZ - newZ;

    double diff_einsum_C2_x = diff.x * from.field.C2xx + diff.y * from.field.C2xy;
    double diff_einsum_C2_y = diff.x * from.field.C2xy + diff.y * from.field.C2yy;

    double diff_einsum_C3_xx = diff.x * from.field.C3xxx + diff.y * from.field.C3xxy;
    double diff_einsum_C3_xy = diff.x * from.field.C3xxy + diff.y * from.field.C3xyy;
    double diff_einsum_C3_yy = diff.x * from.field.C3xyy + diff.y * from.field.C3yyy;

    double diff_einsumx2_C3_x = diff.x * diff_einsum_C3_xx + diff.y * diff_einsum_C3_xy;
    double diff_einsumx2_C3_y = diff.x * diff_einsum_C3_xy + diff.y * diff_einsum_C3_yy;

    to.field += {from.field.C1x + diff_einsum_C2_x + 0.5 * diff_einsumx2_C3_x,
                 from.field.C1y + diff_einsum_C2_y + 0.5 * diff_einsumx2_C3_y,
                 from.field.C2xx + diff_einsum_C3_xx,
                 from.field.C2xy + diff_einsum_C3_xy,
                 from.field.C2yy + diff_einsum_C3_yy,
                 from.field.C3xxx,
                 from.field.C3xxy,
                 from.field.C3xyy,
                 from.field.C3yyy};
}

inline void MF::translate_from_to_point(NodeFMM& from, TsnePoint2D& to)
{
    glm::dvec2 x = to.position;
    glm::dvec2 Z0 = from.COM;
    glm::dvec2 diff = to.position - from.COM;

    double diff_einsum_C2_x = diff.x * from.field.C2xx + diff.y * from.field.C2xy;
    double diff_einsum_C2_y = diff.x * from.field.C2xy + diff.y * from.field.C2yy;

    double diff_einsum_C3_xx = diff.x * from.field.C3xxx + diff.y * from.field.C3xxy;
    double diff_einsum_C3_xy = diff.x * from.field.C3xxy + diff.y * from.field.C3xyy;
    double diff_einsum_C3_yy = diff.x * from.field.C3xyy + diff.y * from.field.C3yyy;

    double diff_einsumx2_C3_x = diff.x * diff_einsum_C3_xx + diff.y * diff_einsum_C3_xy;
    double diff_einsumx2_C3_y = diff.x * diff_einsum_C3_xy + diff.y * diff_einsum_C3_yy;

    to.derivative.x += from.field.C1x + diff_einsum_C2_x + 0.5 * diff_einsumx2_C3_x;
    to.derivative.y += from.field.C1y + diff_einsum_C2_y + 0.5 * diff_einsumx2_C3_y;
}
