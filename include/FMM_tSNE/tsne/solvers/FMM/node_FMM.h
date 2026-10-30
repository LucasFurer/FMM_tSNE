#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>

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

    double C1x{};
    double C1y{};
    double C2xx{};
    double C2xy{};
    double C2yy{};
    double C3xxx{};
    double C3xxy{};
    double C3xyy{};
    double C3yyy{};

    NodeFMM(double in_BB_length, unsigned int in_first_particle_index, unsigned int in_particle_index_amount)
        : BB_length{in_BB_length},
          first_particle_index{in_first_particle_index},
          particle_index_amount{in_particle_index_amount}
    {
    }
};
