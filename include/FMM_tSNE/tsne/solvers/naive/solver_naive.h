#pragma once

#include <FMM_tSNE/tsne/points/tsne_point_2D.h>
#include <FMM_tSNE/tsne/points/vertex_pos2_col3.h>
#include <FMM_tSNE/tsne/solvers/solver.h>
#include <functional>
#include <glm/glm.hpp>
#include <numeric>
#include <thread>
#include <vector>
// #include "../../particles/Particle2D.h"
// #include "../../particles/embeddedPoint.h"
// #include "../../particles/tsnePoint2D.h"
// #include "../../trees/cpu/quadtree.h"
// #include "nBodySolver.h"

template <typename T>
class NBodySolverNaive final : public NBodySolver<T>
{
  public:
    std::function<void(double&, T&, T&)> kernel;

    NBodySolverNaive() = default;

    NBodySolverNaive(std::function<void(double&, T&, T&)> initKernel) { kernel = initKernel; }

    // void solveNbody(double& total, std::vector<T>& points, std::vector<int>& indexTracker) override
    //{
    //     total = 0.0;

    //    if (kernel)
    //    {
    //        for (int i = 0; i < points.size(); i++)
    //        {
    //            for (int j = 0; j < points.size(); j++)
    //            {
    //                if (i != j)
    //                {

    //                    kernel(total, points[i], points[j]);

    //                }
    //            }
    //        }
    //    }
    //}

    void solveNbody(double& total, std::vector<T>& points) override
    {
        total = 0.0;

        const unsigned numThreads = std::thread::hardware_concurrency();
        const size_t chunkSize = (points.size() + numThreads - 1) / numThreads;

        std::vector<std::thread> threads;
        std::vector<double> localTotals(numThreads, 0.0);

        // Thread-local derivative buffers
        std::vector<std::vector<glm::dvec2>> localDerivatives(numThreads,
                                                              std::vector<glm::dvec2>(points.size(), glm::dvec2(0.0)));

        for (unsigned t = 0; t < numThreads; ++t)
        {
            size_t begin = t * chunkSize;
            size_t end = std::min(begin + chunkSize, points.size());

            threads.emplace_back(
                [&, t, begin, end]()
                {
                    double threadTotal = 0.0;

                    for (size_t i = begin; i < end; ++i)
                    {
                        for (size_t j = 0; j < points.size(); ++j)
                        {
                            if (i == j) continue;

                            T sink = points[i];
                            T source = points[j];

                            kernel(threadTotal, sink, source);

                            localDerivatives[t][i] += sink.derivative;
                            // localDerivatives[t][j] += source.derivative;
                        }
                    }

                    localTotals[t] = threadTotal;
                });
        }

        for (std::thread& th : threads)
            th.join();

        // Reduce totals
        total = std::accumulate(localTotals.begin(), localTotals.end(), 0.0);

        // Reduce derivatives
        for (size_t t = 0; t < numThreads; ++t)
            for (size_t i = 0; i < points.size(); ++i)
                points[i].derivative += localDerivatives[t][i];
    }

    void updateTree(std::vector<T>& points, glm::dvec2 minPos, glm::dvec2 maxPos) override {}

    std::vector<VertexPos2Col3> getNodesBufferData(int level) override { return std::vector<VertexPos2Col3>(); }

  private:
};

inline void TSNEnaiveKernel(double& total, TsnePoint2D& sinkPoint, TsnePoint2D& sourcePoint)
{
    // glm::vec2 diff = sinkPoint.position - sourcePoint.position;
    // float dist = glm::length(diff);

    // float forceDecay = 1.0f / (1.0f + (dist * dist));
    // total += static_cast<double>(2.0f * forceDecay);

    // sinkPoint.derivative += forceDecay * forceDecay * diff;
    // sourcePoint.derivative += forceDecay * forceDecay * -diff;

    glm::dvec2 diff = sinkPoint.position - sourcePoint.position;
    double dist = glm::length(diff);

    double forceDecay = 1.0 / (1.0 + (dist * dist));
    total += forceDecay;

    sinkPoint.derivative += forceDecay * forceDecay * diff;
}
