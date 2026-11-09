#pragma once

#include <FMM_tSNE/tsne/tensors/MF.h>
#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>

struct NodeSFMM
{
    glm::dvec2 COM{};

    double M0{};
    double M2xx{};
    double M2xy{};
    double M2yy{};

    double BB_length;

    unsigned int first_point_index; // first_particle_index + particle_index_amount is index of last particle
    unsigned int point_count;       // amount of particles in node

    unsigned int first_child_index; // first_child_index + child_index_amount is index of last child
    unsigned char child_count;      // amount of children (max is 4)

    MF field{};

    NodeSFMM(double in_BB_length, unsigned int in_first_particle_index, unsigned int in_particle_index_amount)
        : BB_length{in_BB_length},
          first_point_index{in_first_particle_index},
          point_count{in_particle_index_amount}
    {
    }
};
