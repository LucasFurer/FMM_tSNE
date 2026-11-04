#pragma once

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
};
