#pragma once

#include <FMM_tSNE/tsne/tsne.h>

#include <GLFW/glfw3.h>
#include <cmath>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <glm/glm.hpp>
// #include <glm/gtx/component_wise.hpp>
// #include <glm/gtx/string_cast.hpp>
#include <FMM_tSNE/opengl_interaction/buffer.h>
#include <iostream>
#include <limits>
#include <map>
#include <numbers>
#include <string>
#include <unsupported/Eigen/SparseExtra>
#include <utility>
#include <vector>

// #include "../Timer.h"
// #include "../common.h"
// #include "../dataLoaders/loader.h"
// #include "../ffthelper.h"
// #include "../nbodysolvers/cpu/nBodySolverBH.h"
// #include "../nbodysolvers/cpu/nBodySolverBHMP.h"
// #include "../nbodysolvers/cpu/nBodySolverBHR.h"
// #include "../nbodysolvers/cpu/nBodySolverBHRMP.h"
// #include "../nbodysolvers/cpu/nBodySolverFMM.h"
// #include "../nbodysolvers/cpu/nBodySolverFMM_MORTON.h"
// #include "../nbodysolvers/cpu/nBodySolverFMM_SYM_MORTON.h"
// #include "../nbodysolvers/cpu/nBodySolverFMMiter.h"
// #include "../nbodysolvers/cpu/nBodySolverNaive.h"
// #include "../nbodysolvers/cpu/nBodySolverPM.h"
// #include "../openGLhelper/buffer.h"
// #include "../particleMesh.h"
// #include "../particles/embeddedPoint.h"
// #include "../particles/tsnePoint2D.h"
// #include "tsne.h"

class TSNE_buffers final : public TSNE
{
  public:
    Buffer* embeddedBuffer; // todo: add = nullptr here
    Buffer* nodeBuffer;
    float forceSize{1.0f};
    Buffer* forceBuffer;

    int follow{1};
    int nodeLevelToShow{0};

    float desired_iteration_per_second{0.0f}; // limits the speed of tsne
    double time_since_last_iteration{0.0};

    TSNE_buffers() = default;

    TSNE_buffers(double init_min_theta,
                 double init_max_theta,
                 double init_cell_size,
                 std::string dataSet,
                 int data_amount,
                 float perplexity,
                 unsigned int seed)
        : TSNE(init_min_theta, init_max_theta, init_cell_size, dataSet, data_amount, perplexity, seed)
    {
        resetBuffers();
    }

    ~TSNE_buffers() override
    {
        delete embeddedBuffer;
        delete nodeBuffer;
        delete forceBuffer;
    }

    void resetBuffers()
    {
#ifdef INDEX_TRACKER
        embeddedBuffer = new Buffer(embeddedPoints, DataLayout::Double2_Double2_Int1_Int1_Int32t1, GL_DYNAMIC_DRAW);
#else
        embeddedBuffer = new Buffer(embeddedPoints, DataLayout::Double2_Double2_Int1, GL_DYNAMIC_DRAW);
#endif

        std::vector<VertexPos2Col3> nodesBufferData = nBodySolvers[nBodySelect]->getNodesBufferData(nodeLevelToShow);
        nodeBuffer = new Buffer(nodesBufferData, DataLayout::Float2_Float3, GL_DYNAMIC_DRAW);

        std::vector<VertexPos2Col3> forceLines =
            VertexPos2Col3::particlesAccelerationsToVertexPos2Col3(embeddedPoints, forceSize);
        forceBuffer = new Buffer(forceLines, DataLayout::Float2_Float3, GL_DYNAMIC_DRAW);
    }

    void updateBuffers()
    {
#ifdef INDEX_TRACKER
        embeddedBuffer->update_vertex_buffer(embeddedPoints, DataLayout::Double2_Double2_Int1_Int1_Int32t1);
#else
        embeddedBuffer->update_vertex_buffer(embeddedPoints, DataLayout::Double2_Double2_Int1);
#endif

        std::vector<VertexPos2Col3> nodesBufferData = nBodySolvers[nBodySelect]->getNodesBufferData(nodeLevelToShow);
        nodeBuffer->update_vertex_buffer(nodesBufferData, DataLayout::Float2_Float3);

        std::vector<VertexPos2Col3> forceLines =
            VertexPos2Col3::particlesAccelerationsToVertexPos2Col3(embeddedPoints, forceSize);
        forceBuffer->update_vertex_buffer(forceLines, DataLayout::Float2_Float3);
    }

    void timeStep()
    {
        if (iteration_counter == 1)
        {
            // costFunction(this->embeddedPoints, this->Pmatrix);
            thousand_iteration_timer.startTimer();
        }

        if (iteration_counter == 1000 && !reached_thousand_iterations)
        {
            reached_thousand_iterations = true;
            desired_iteration_per_second = 0.0f;
            thousand_iteration_timer.endTimer("A thousand iterations");

            costFunction(this->embeddedPoints, this->Pmatrix);
        }

        if (glfwGetTime() - time_since_last_iteration >= 1.0 / static_cast<double>(desired_iteration_per_second))
        {
            time_since_last_iteration = glfwGetTime();

            // std::cout << "------------------------------------\n";
            Timer time_step_timer;

            {
                Timer derivative_timer;
                updateDerivative();
                derivative_timer.endTimer("__updateDerivative");

                Timer update_timer;
                updatePoints();
                update_timer.endTimer("__updatePoints");

                Timer time_update_tree;
                nBodySolvers[nBodySelect]->updateTree(embeddedPoints, minPos, maxPos);
                time_update_tree.endTimer("__update tree");

                updateBuffers();
            }

            time_step_timer.endTimer("timeStep");
        }
    }
};
