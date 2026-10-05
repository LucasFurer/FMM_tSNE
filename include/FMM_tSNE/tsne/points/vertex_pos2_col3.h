#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <vector>
// #include <glm/gtx/string_cast.hpp>

struct VertexPos2Col3
{
    glm::vec2 position;
    glm::vec3 color;

    VertexPos2Col3(glm::vec2 initPosition, glm::vec3 initColor)
    {
        position = initPosition;
        color = initColor;
    }

    template <typename T>
    static std::vector<VertexPos2Col3> particlesToVertexPos2Col3(const std::vector<T>& particles, float forceSize)
    {
        std::vector<VertexPos2Col3> result;
        result.reserve(particles.size());
        for (int i = 0; i < particles.size(); i++)
        {
            glm::vec2 linePosB = particles[i].position;
            glm::vec3 lineColB = glm::vec3(1.0f, 0.0f, 0.0f);

            glm::vec2 linePosE = particles[i].position + forceSize * forceSize * particles[i].derivative;
            // glm::vec2 linePosE = particles[i].position + particles[i].speed;
            glm::vec3 lineColE = glm::vec3(1.0f, 0.0f, 0.0f);

            result.push_back(VertexPos2Col3(linePosB, lineColB));
            result.push_back(VertexPos2Col3(linePosE, lineColE));
        }
        return result;
    }

    template <typename T>
    static std::vector<VertexPos2Col3> particlesAccelerationsToVertexPos2Col3(const std::vector<T>& points,
                                                                              float forceSize)
    {
        std::vector<VertexPos2Col3> result;
        result.reserve(points.size() * 2);
        for (int i = 0; i < points.size(); i++)
        {
            glm::vec2 linePosB = glm::vec2(points[i].position.x, points[i].position.y);
            glm::vec3 lineColB = glm::vec3(1.0f, 0.0f, 0.0f);

            glm::vec2 linePosE = points[i].position + static_cast<double>(forceSize * forceSize) * points[i].derivative;
            // glm::vec2 linePosE = particles[i].position + particles[i].speed;
            glm::vec3 lineColE = glm::vec3(1.0f, 0.0f, 0.0f);

            result.push_back(VertexPos2Col3(linePosB, lineColB));
            result.push_back(VertexPos2Col3(linePosE, lineColE));
        }
        return result;
    }
};
