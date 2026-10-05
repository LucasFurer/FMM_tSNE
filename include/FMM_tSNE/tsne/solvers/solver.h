#pragma once

#include <FMM_tSNE/opengl_interaction/buffer.h>
#include <FMM_tSNE/tsne/points/vertex_pos2_col3.h>
#include <glm/glm.hpp>
#include <vector>

template <typename T>
class NBodySolver
{
  public:
    int maxChildren;
    double theta;
    double cell_size = 0.0;

    virtual void solveNbody(double& total, std::vector<T>& points) = 0;
    virtual void updateTree(std::vector<T>& points, glm::dvec2 minPos, glm::dvec2 maxPos) = 0;
    virtual std::vector<VertexPos2Col3> getNodesBufferData(int level) = 0;

    virtual ~NBodySolver() = default;
};
