#pragma once

#include <FMM_tSNE/tsne/points/tsne_point_2D.h>
#include <FMM_tSNE/tsne/points/vertex_pos2_col3.h>
#include <FMM_tSNE/tsne/solvers/naive_GPU/solver_naive_GPU_backend.h>
#include <FMM_tSNE/tsne/solvers/solver.h>
#include <glm/glm.hpp>
#include <vector>

// ---------------------------------------------------------------------------
// Drop-in replacement for NBodySolverNaive<T>, running on the GPU.
// T must expose `.position` and `.derivative` (glm::dvec2).
// ---------------------------------------------------------------------------
template <typename T>
class NBodySolverNaiveCUDA final : public NBodySolver<T>
{
  public:
    NBodySolverNaiveCUDA() = default;

    void solveNbody(double& total, std::vector<T>& points) override
    {
        total = 0.0;
        const int n = static_cast<int>(points.size());
        if (n == 0) return;

        hostPos.resize(2 * static_cast<size_t>(n));
        hostDeriv.resize(2 * static_cast<size_t>(n));

        for (int i = 0; i < n; i++)
        {
            hostPos[2 * i + 0] = points[i].position.x;
            hostPos[2 * i + 1] = points[i].position.y;
        }

        backend.solve(hostPos.data(), hostDeriv.data(), n, total);

        for (int i = 0; i < n; i++)
        {
            points[i].derivative.x += hostDeriv[2 * i + 0];
            points[i].derivative.y += hostDeriv[2 * i + 1];
        }
    }

    void updateTree(std::vector<T>& points, glm::dvec2 minPos, glm::dvec2 maxPos) override {}

    std::vector<VertexPos2Col3> getNodesBufferData(int level) override { return std::vector<VertexPos2Col3>(); }

  private:
    NaiveCudaBackend backend;
    std::vector<double> hostPos; // reused staging buffers
    std::vector<double> hostDeriv;
};
