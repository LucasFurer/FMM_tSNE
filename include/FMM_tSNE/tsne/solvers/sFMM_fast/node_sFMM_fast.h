#pragma once

#include <Fastor/Fastor.h>
#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>

struct NodeSFMMFast
{
    glm::dvec2 COM;

    double M0;
    Fastor::Tensor<double, 2, 2> M2;

    Fastor::Tensor<double, 2> C1;
    Fastor::Tensor<double, 2, 2> C2;
    Fastor::Tensor<double, 2, 2, 2> C3;

    glm::dvec2 BB_centre;
    double BB_length;

    unsigned int first_particle_index;  // firstParticleIndex + particleIndexAmount is index of last particle
    unsigned int particle_index_amount; // amount of particles in node

    unsigned int first_child_index;   // firstChildIndex + 3 is index of last child
    unsigned char child_index_amount; // amount of children (max is 4)
};
