#pragma once

#include "FMM_tSNE/tsne/points/tsne_point_2D.h"
#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>

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
    // double C1x{};
    // double C1y{};
    // double C2xx{};
    // double C2xy{};
    // double C2yy{};
    // double C3xxx{};
    // double C3xxy{};
    // double C3xyy{};
    // double C3yyy{};

    NodeFMM(double in_BB_length, unsigned int in_first_particle_index, unsigned int in_particle_index_amount)
        : BB_length{in_BB_length},
          first_particle_index{in_first_particle_index},
          particle_index_amount{in_particle_index_amount}
    {
    }

    void normalize()
    {
        double norm_val = 1.0 / M0;
        // we dont normalize C1
        field.C2xx *= norm_val;
        field.C2xy *= norm_val;
        field.C2yy *= norm_val;
        field.C3xxx *= norm_val;
        field.C3xxy *= norm_val;
        field.C3xyy *= norm_val;
        field.C3yyy *= norm_val;
    }

    void translate_to_node(NodeFMM& to)
    {
        glm::dvec2 oldZ = to.COM;
        glm::dvec2 newZ = COM;
        glm::dvec2 diff = oldZ - newZ;

        double diff_einsum_C2_x = diff.x * field.C2xx + diff.y * field.C2xy;
        double diff_einsum_C2_y = diff.x * field.C2xy + diff.y * field.C2yy;

        double diff_einsum_C3_xx = diff.x * field.C3xxx + diff.y * field.C3xxy;
        double diff_einsum_C3_xy = diff.x * field.C3xxy + diff.y * field.C3xyy;
        double diff_einsum_C3_yy = diff.x * field.C3xyy + diff.y * field.C3yyy;

        double diff_einsumx2_C3_x = diff.x * diff_einsum_C3_xx + diff.y * diff_einsum_C3_xy;
        double diff_einsumx2_C3_y = diff.x * diff_einsum_C3_xy + diff.y * diff_einsum_C3_yy;

        to.field += {field.C1x + diff_einsum_C2_x + 0.5 * diff_einsumx2_C3_x,
                     field.C1y + diff_einsum_C2_y + 0.5 * diff_einsumx2_C3_y,
                     field.C2xx + diff_einsum_C3_xx,
                     field.C2xy + diff_einsum_C3_xy,
                     field.C2yy + diff_einsum_C3_yy,
                     field.C3xxx,
                     field.C3xxy,
                     field.C3xyy,
                     field.C3yyy};
    }
    void translate_to_point(TsnePoint2D& to)
    {
        glm::dvec2 x = to.position;
        glm::dvec2 Z0 = COM;
        glm::dvec2 diff = to.position - COM;

        double diff_einsum_C2_x = diff.x * field.C2xx + diff.y * field.C2xy;
        double diff_einsum_C2_y = diff.x * field.C2xy + diff.y * field.C2yy;

        double diff_einsum_C3_xx = diff.x * field.C3xxx + diff.y * field.C3xxy;
        double diff_einsum_C3_xy = diff.x * field.C3xxy + diff.y * field.C3xyy;
        double diff_einsum_C3_yy = diff.x * field.C3xyy + diff.y * field.C3yyy;

        double diff_einsumx2_C3_x = diff.x * diff_einsum_C3_xx + diff.y * diff_einsum_C3_xy;
        double diff_einsumx2_C3_y = diff.x * diff_einsum_C3_xy + diff.y * diff_einsum_C3_yy;

        to.derivative.x += field.C1x + diff_einsum_C2_x + 0.5 * diff_einsumx2_C3_x;
        to.derivative.y += field.C1y + diff_einsum_C2_y + 0.5 * diff_einsumx2_C3_y;
    }
};
