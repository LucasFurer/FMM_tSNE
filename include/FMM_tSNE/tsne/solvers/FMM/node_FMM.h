#pragma once

#include <FMM_tSNE/tsne/tensors/MF.h>
#include <glm/ext/vector_double2.hpp>
#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>

// alignas(64) makes each node occupy their own cache line
struct alignas(64) NodeFMM
{
    glm::dvec2 COM{};

    double M2xx{};
    double M2xy{};
    double M2yy{};

    double BB_length;

    unsigned int first_particle_index; // first_particle_index + particle_index_amount is index of last particle
    union
    {
        unsigned int particle_index_amount; // amount of particles in node
        unsigned int M0;                    // total mass of the node
    };

    unsigned int first_child_index;   // first_child_index + child_index_amount is index of last child
    unsigned char child_index_amount; // amount of children (max is 4)

    NodeFMM(double in_BB_length, unsigned int in_first_particle_index, unsigned int in_particle_index_amount)
        : BB_length{in_BB_length},
          first_particle_index{in_first_particle_index},
          particle_index_amount{in_particle_index_amount}
    {
    }
};
