#pragma once

#include <GLFW/glfw3.h>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <glm/glm.hpp>
// #include <glm/gtx/component_wise.hpp>
// #include <glm/gtx/string_cast.hpp>
#include <FMM_tSNE/data_loaders/MNIST_data.h>
#include <FMM_tSNE/data_loaders/sparse_entry_COO_2D.h>
#include <FMM_tSNE/data_loaders/sparse_entry_CSC_2D.h>
#include <FMM_tSNE/opengl_interaction/scene.h>
#include <FMM_tSNE/timer/timer.h>
#include <FMM_tSNE/tsne/points/tsne_point_2D.h>
#include <FMM_tSNE/tsne/solvers/solver.h>
#include <FMM_tSNE/tsne/solvers/solver_naive.h>
#include <iostream>
#include <limits>
#include <map>
#include <numbers>
#include <random>
#include <string>
#include <unsupported/Eigen/SparseExtra>
#include <utility>
#include <vector>

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
// #include "../nbodysolvers/cpu/nBodySolverTest.h"
// #include "../openGLhelper/buffer.h"
// #include "../particleMesh.h"
// #include "../particles/embeddedPoint.h"
// #include "../particles/tsnePoint2D.h"

class TSNE
{
  public:
    std::vector<TsnePoint2D> embeddedPoints;
    std::vector<TsnePoint2D> embeddedPointsPrev;
    std::vector<TsnePoint2D> embeddedPointsPrevPrev;

    glm::dvec2 minPos{std::numeric_limits<double>::max()};
    glm::dvec2 maxPos{std::numeric_limits<double>::lowest()};

    std::map<std::string, NBodySolver<TsnePoint2D>*> nBodySolvers;
    std::string nBodySelect{"naive"}; // default selector

    double learnRate{0.0};

    std::vector<uint8_t> labels;
    Eigen::SparseMatrix<double> Pmatrix;
    std::vector<unsigned int> row_P;
    std::vector<unsigned int> col_P;
    std::vector<double> val_P;

    int iteration_counter{0}; // keeps track of which iteration we are on

    Timer thousand_iteration_timer;
    bool reached_thousand_iterations{false};

    double min_theta{0.0};
    double max_theta{0.0};

    double cell_size{1.0};

    TSNE() = default;

    TSNE(double init_min_theta,
         double init_max_theta,
         double init_cell_size,
         std::string dataSet,
         int data_amount,
         float perplexity,
         unsigned int seed)
    {
        min_theta = init_min_theta;
        max_theta = init_max_theta;
        cell_size = init_cell_size;

        learnRate = static_cast<double>(data_amount) / 15.0;

        std::filesystem::path labelsPath =
            std::filesystem::current_path() /
            ("data/" + dataSet + "/" + std::to_string(data_amount) + "/label_amount" + std::to_string(data_amount) +
             "_perp" + std::to_string((int) perplexity) + ".bin");
        std::filesystem::path fileName =
            std::filesystem::current_path() /
            ("data/" + dataSet + "/" + std::to_string(data_amount) + "/P_matrix_amount" + std::to_string(data_amount) +
             "_perp" + std::to_string((int) perplexity) + ".mtx");
        labels = MNIST_data::loadLabels(labelsPath.string());
        Pmatrix = MNIST_data::loadPmatrix(fileName.string());

        generateInitialPoints(data_amount, seed, false, 2.0);

        setupEfficientPmatrix();

        updateMinMaxPos();

        int max_children_per_node = 16;
        nBodySolvers["naive"] = new NBodySolverNaive<TsnePoint2D>(&TSNEnaiveKernel);
        // nBodySolvers["BH"] =
        //     new NBodySolverBH<TsnePoint2D>(&TSNEBHPNKernel, &TSNEBHPPKernel, max_children_per_node, max_theta);
        // nBodySolvers["BH"]->updateTree(embeddedPoints, minPos, maxPos);
        // nBodySolvers["BHR"] =
        //     new NBodySolverBHR<TsnePoint2D>(&TSNEBHRNPKernel, &TSNEBHRPPKernel, max_children_per_node, max_theta);
        // nBodySolvers["BHR"]->updateTree(embeddedPoints, minPos, maxPos);
        // nBodySolvers["BHMP"] =
        //     new NBodySolverBHMP<TsnePoint2D>(&TSNEBHMPPNKernel, &TSNEBHMPPPKernel, max_children_per_node, max_theta);
        // nBodySolvers["BHMP"]->updateTree(embeddedPoints, minPos, maxPos);
        // nBodySolvers["BHRMP"] =
        //     new NBodySolverBHRMP<TsnePoint2D>(&TSNEBHRMPNPKernel, &TSNEBHRMPPPKernel, max_children_per_node,
        //     max_theta);
        // nBodySolvers["BHRMP"]->updateTree(embeddedPoints, minPos, maxPos);
        // nBodySolvers["FMM"] = new NBodySolverFMM<TsnePoint2D>(
        //     &TSNEFMMNNKernel, &TSNEFMMPNKernel, &TSNEFMMNPKernel, &TSNEFMMPPKernel, max_children_per_node,
        //     max_theta);
        // nBodySolvers["FMM"]->updateTree(embeddedPoints, minPos, maxPos);
        // nBodySolvers["PM"] = new NBodySolverPM<TsnePoint2D>(Pmatrix, embeddedPoints, 4, cell_size, 4);
        // nBodySolvers["PM"]->updateTree(embeddedPoints, minPos, maxPos);
#ifdef INDEX_TRACKER
        nBodySolvers["FMM_MORTON"] = new NBodySolverFMM_MORTON<TsnePoint2D>(
            &TSNEFMM_MORTONNNKernel,
            &TSNEFMM_MORTONPNKernel,
            &TSNEFMM_MORTONNPKernel,
            &TSNEFMM_MORTONPPKernel,
            max_children_per_node,
            NBodySolverFMM_MORTON<TsnePoint2D>::getDepth(max_children_per_node, data_amount),
            max_theta);
        nBodySolvers["FMM_MORTON"]->updateTree(embeddedPoints, minPos, maxPos);
        nBodySolvers["FMM_SYM_MORTON"] = new NBodySolverFMM_SYM_MORTON<TsnePoint2D>(
            &TSNE_FMM_SYM_MORTON_NN_Kernel,
            &TSNE_FMM_SYM_MORTON_PN_Kernel,
            &TSNE_FMM_SYM_MORTON_PP_Kernel,
            max_children_per_node,
            NBodySolverFMM_SYM_MORTON<TsnePoint2D>::getDepth(max_children_per_node, data_amount),
            max_theta);
        nBodySolvers["FMM_SYM_MORTON"]->updateTree(embeddedPoints, minPos, maxPos);
#endif
    }

    virtual ~TSNE()
    {
        for (std::pair<const std::string, NBodySolver<TsnePoint2D>*> nBodySolverPointer : nBodySolvers)
        {
            delete nBodySolverPointer.second;
        }

        nBodySolvers.clear();
    }

    void resetTsne(double init_min_theta,
                   double init_max_theta,
                   double init_cell_size,
                   std::string dataSet,
                   int data_amount,
                   float perplexity,
                   unsigned int seed)
    {
        iteration_counter = 0;

        min_theta = init_min_theta;
        max_theta = init_max_theta;
        cell_size = init_cell_size;

        learnRate = static_cast<double>(data_amount) / 15.0;

        std::filesystem::path labelsPath =
            std::filesystem::current_path() /
            ("data/" + dataSet + "/" + std::to_string(data_amount) + "/label_amount" + std::to_string(data_amount) +
             "_perp" + std::to_string((int) perplexity) + ".bin");
        std::filesystem::path fileName =
            std::filesystem::current_path() /
            ("data/" + dataSet + "/" + std::to_string(data_amount) + "/P_matrix_amount" + std::to_string(data_amount) +
             "_perp" + std::to_string((int) perplexity) + ".mtx");
        labels = MNIST_data::loadLabels(labelsPath.string());
        Pmatrix = MNIST_data::loadPmatrix(fileName.string());

        generateInitialPoints(data_amount, seed, false, 2.0);

        setupEfficientPmatrix();

        updateMinMaxPos();

        setThetaForAll(min_theta, cell_size);
        nBodySolvers["BH"]->updateTree(embeddedPoints, minPos, maxPos);
        nBodySolvers["BHR"]->updateTree(embeddedPoints, minPos, maxPos);
        nBodySolvers["BHMP"]->updateTree(embeddedPoints, minPos, maxPos);
        nBodySolvers["BHRMP"]->updateTree(embeddedPoints, minPos, maxPos);
        nBodySolvers["FMM"]->updateTree(embeddedPoints, minPos, maxPos);
        nBodySolvers["PM"]->updateTree(embeddedPoints, minPos, maxPos);
#ifdef INDEX_TRACKER
        nBodySolvers["FMM_MORTON"]->updateTree(embeddedPoints, minPos, maxPos);
        nBodySolvers["FMM_SYM_MORTON"]->updateTree(embeddedPoints, minPos, maxPos);
#endif
    }

    void updatePoints()
    {
        embeddedPointsPrev.swap(embeddedPointsPrevPrev);
        embeddedPoints.swap(embeddedPointsPrev);

        thetaFunction();

        double accelerationRate = iteration_counter < 250 ? 0.5 : 0.8;

        for (int i = 0; i < embeddedPoints.size(); i++)
        {
#ifdef INDEX_TRACKER
            embeddedPoints[i].derivative = embeddedPointsPrev[i].derivative;
            embeddedPoints[i].label = embeddedPointsPrev[i].label;
            embeddedPoints[i].ID = embeddedPointsPrev[i].ID;
            embeddedPoints[i].position =
                embeddedPointsPrev[i].position + learnRate * embeddedPointsPrev[i].derivative +
                accelerationRate * (embeddedPointsPrev[i].position - embeddedPointsPrevPrev[i].position);
#else
            embeddedPoints[i].derivative = embeddedPointsPrev[i].derivative;
            embeddedPoints[i].label = embeddedPointsPrev[i].label;
            embeddedPoints[i].position =
                embeddedPointsPrev[i].position + learnRate * embeddedPointsPrev[i].derivative +
                accelerationRate * (embeddedPointsPrev[i].position - embeddedPointsPrevPrev[i].position);
#endif
        }

        updateMinMaxPos();

        iteration_counter++;
    }

    void thetaFunction()
    {
        // double falloff_strength = 1.0f;
        // double theta_result =
        //     (max_theta - min_theta) *
        //     std::max(1.0 - std::pow(static_cast<double>(iteration_counter) / 999.0, falloff_strength), 0.0) +
        //     min_theta;

        double sharpness = 1.0;
        double extra_theta_ratio =
            0.5 + 0.5 * std::cos(std::numbers::pi *
                                 std::min(std::pow(static_cast<double>(iteration_counter) / 999.0, sharpness), 1.0));
        double theta_diff = max_theta - min_theta;
        double theta_result = min_theta + extra_theta_ratio * theta_diff;

        std::cout << "set theta to: " << theta_result << std::endl;
        std::cout << "set cell size to: " << cell_size << std::endl;

        setThetaForAll(theta_result, cell_size);
    }

    void setMinMaxTheta(double set_min_theta, double set_max_theta)
    {
        min_theta = set_min_theta;
        max_theta = set_max_theta;
    }

    void setThetaForAll(double new_theta, double new_cell_size)
    {
        for (std::pair<const std::string, NBodySolver<TsnePoint2D>*> nBodySolverPointer : nBodySolvers)
        {
            nBodySolverPointer.second->theta = new_theta;
            nBodySolverPointer.second->cell_size = new_cell_size;
        }
    }

    void updateAllTrees()
    {
        updateMinMaxPos();

        for (std::pair<const std::string, NBodySolver<TsnePoint2D>*> nBodySolverPointer : nBodySolvers)
        {
            nBodySolverPointer.second->updateTree(embeddedPoints, minPos, maxPos);
        }
    }

    void updateMinMaxPos()
    {
        minPos = glm::dvec2(std::numeric_limits<double>::max());
        maxPos = glm::dvec2(std::numeric_limits<double>::lowest());

        for (TsnePoint2D& point : embeddedPoints)
        {
            minPos = glm::min(minPos, point.position);
            maxPos = glm::max(maxPos, point.position);
        }
    }

    void updateDerivative()
    {
        resetDeriv();

        updateRepulsive();

        updateAttractive();
    }

    void resetDeriv()
    {
        for (TsnePoint2D& embeddedPoint : embeddedPoints)
            embeddedPoint.derivative = glm::dvec2(0.0);
    }

    void updateRepulsive()
    {
        double QijTotal = 0.0;
        nBodySolvers[nBodySelect]->solveNbody(QijTotal, embeddedPoints);
        QijTotal = 4.0 / QijTotal;

        for (int i = 0; i < embeddedPoints.size(); i++)
            embeddedPoints[i].derivative *= QijTotal;

#ifdef INDEX_TRACKER
        for (size_t i = 0; i < embeddedPointsPrevPrev.size(); i++)
            embeddedPointsPrevPrev[embeddedPoints[i].ID] = embeddedPoints[i];
        embeddedPointsPrevPrev.swap(embeddedPoints);
#endif
    }

    void updateAttractiveOUTDATED1()
    {
        float exaggeration = iteration_counter < 250 ? 16.0f : 4.0f;

        for (int k = 0; k < Pmatrix.outerSize();
             ++k) // https://stackoverflow.com/questions/22421244/eigen-package-iterate-over-row-major-sparse-matrix
        {
            for (Eigen::SparseMatrix<double>::InnerIterator it(Pmatrix, k); it; ++it)
            {
                TsnePoint2D& pointR = embeddedPoints[it.row()];
                TsnePoint2D& pointC = embeddedPoints[it.col()];

                glm::vec2 diff = pointR.position - pointC.position;
                float dist = diff.x * diff.x + diff.y * diff.y;

                pointC.derivative += exaggeration * static_cast<float>(it.value()) * (diff / (1.0f + dist));
            }
        }
    }

    void updateAttractiveOUTDATED2()
    {
        double exageration = iteration_counter < 250 ? 16.0 : 4.0; // the early exaggeration is 4.0f

        for (int n = 0; n < embeddedPoints.size(); n++)
        {
            // #ifdef INDEX_TRACKER
            // TsnePoint2D& pointR = embeddedPoints[indexTracker[n]];
            // #else
            TsnePoint2D& pointR = embeddedPoints[n];
            // #endif

            glm::dvec2 dim{0.0f};

            for (unsigned int i = row_P[n]; i < row_P[n + 1]; i++)
            {
                unsigned int col = col_P[i];
                // #ifdef INDEX_TRACKER
                // TsnePoint2D& pointC = embeddedPoints[indexTracker[col]];
                // #else
                TsnePoint2D& pointC = embeddedPoints[col];
                // #endif

                glm::dvec2 diff = pointC.position - pointR.position;
                double d_ij = diff.x * diff.x + diff.y * diff.y;
                double q_ij = 1.0 / (1.0 + d_ij);

                dim += exageration * val_P[i] * q_ij * diff;
            }

            pointR.derivative += dim;
        }
    }

    void updateAttractive()
    {
        double exageration = iteration_counter < 250 ? 16.0 : 4.0; // the early exaggeration is 4.0f
        // double exageration = iteration_counter < 250 ? 4.0 : 4.0; // the early exaggeration is 4.0f

        for (int n = 0; n < embeddedPoints.size(); n++)
        {
            TsnePoint2D& pointR = embeddedPoints[n];

            glm::dvec2 dim{0.0f};

            for (unsigned int i = row_P[n]; i < row_P[n + 1]; i++)
            {
                unsigned int col = col_P[i];
                TsnePoint2D& pointC = embeddedPoints[col];

                glm::dvec2 diff = pointC.position - pointR.position;
                double d_ij = diff.x * diff.x + diff.y * diff.y;
                double q_ij = 1.0 / (1.0 + d_ij);

                dim += exageration * val_P[i] * q_ij * diff;
            }

            pointR.derivative += dim;
        }
    }

    void costFunction(const std::vector<TsnePoint2D>& points, Eigen::SparseMatrix<double>& Pmatrix)
    {
#ifdef INDEX_TRACKER
        for (size_t i = 0; i < embeddedPointsPrevPrev.size(); i++)
            embeddedPointsPrevPrev[embeddedPoints[i].ID] = embeddedPoints[i];
        embeddedPointsPrevPrev.swap(embeddedPoints);
#endif

        double QijTotal = 0.0;

        const unsigned numThreads = std::thread::hardware_concurrency();
        const size_t chunkSize = (points.size() + numThreads - 1) / numThreads;

        std::vector<std::thread> threads;
        std::vector<double> localTotals(numThreads, 0.0);

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

                            const TsnePoint2D& point_i = points[i];
                            const TsnePoint2D& point_j = points[j];

                            glm::dvec2 diff = point_j.position - point_i.position;
                            double distance_squared = diff.x * diff.x + diff.y * diff.y;
                            localTotals[t] += 1.0 / (1.0 + distance_squared);
                        }
                    }
                });
        }

        for (std::thread& th : threads)
            th.join();

        QijTotal = std::accumulate(localTotals.begin(), localTotals.end(), 0.0);

        double totalCost = 0.0;
        for (int k = 0; k < Pmatrix.outerSize();
             ++k) // https://stackoverflow.com/questions/22421244/eigen-package-iterate-over-row-major-sparse-matrix
        {
            for (Eigen::SparseMatrix<double>::InnerIterator it(Pmatrix, k); it; ++it)
            {
                if (it.col() != it.row())
                {
                    const TsnePoint2D& point_col = points[it.col()];
                    const TsnePoint2D& point_row = points[it.row()];

                    glm::dvec2 diff = point_col.position - point_row.position;
                    double distance_squared = diff.x * diff.x + diff.y * diff.y;
                    double Qij = (1.0 / (1.0 + distance_squared)) / QijTotal;

                    double Pij = it.value();

                    totalCost += Pij * std::log2(Pij / Qij);
                }
            }
        }

        std::cout << "totalCost: " << totalCost << std::endl;

#ifdef INDEX_TRACKER
        nBodySolvers[nBodySelect]->updateTree(embeddedPoints, minPos, maxPos);
#endif
    }

    void setupEfficientPmatrix()
    {
        // const double* values = Pmatrix.valuePtr();
        // const int* innerIndices = Pmatrix.innerIndexPtr();
        // const int* outerStarts = Pmatrix.outerIndexPtr();
        // int nnz = Pmatrix.nonZeros();
        // int rows = Pmatrix.rows();

        // val_P.assign(values, values + nnz);
        // col_P.assign(innerIndices, innerIndices + nnz);
        // row_P.assign(outerStarts, outerStarts + rows + 1);

        std::size_t N = embeddedPoints.size();

        row_P = std::vector<unsigned int>(N + 1, 0u);
        col_P = std::vector<unsigned int>(Pmatrix.nonZeros(), 0u);
        val_P = std::vector<double>(Pmatrix.nonZeros(), 0.0);

        int PmatrixCounter = 0;

        std::vector<SparseEntryCOO2D> sparse_matrix_COO(Pmatrix.nonZeros());
        for (int k = 0; k < Pmatrix.outerSize();
             ++k) // https://stackoverflow.com/questions/22421244/eigen-package-iterate-over-row-major-sparse-matrix
        {
            for (Eigen::SparseMatrix<double>::InnerIterator it(Pmatrix, k); it; ++it)
            {
                sparse_matrix_COO[PmatrixCounter] = SparseEntryCOO2D(it.col(), it.row(), it.value());

                PmatrixCounter++;
            }
        }
        std::sort(sparse_matrix_COO.begin(),
                  sparse_matrix_COO.end(),
                  [](const SparseEntryCOO2D& a, const SparseEntryCOO2D& b) -> bool
                  {
                      if (a.row != b.row)
                          return a.row < b.row;
                      else
                          return a.col < b.col;
                  });
        row_P[0] = 0;
        int entryCounter = 0;
        for (int r = 0; r < N; r++) // go over each row
        {
            int amount_in_row = 0;

            while (entryCounter < Pmatrix.nonZeros() && sparse_matrix_COO[entryCounter].row == r)
            {
                col_P[entryCounter] = sparse_matrix_COO[entryCounter].col;
                val_P[entryCounter] = sparse_matrix_COO[entryCounter].val;
                amount_in_row++;
                entryCounter++;
            }

            row_P[r + 1] = row_P[r] + amount_in_row;
        }
    }

    void generateInitialPoints(int data_amount, unsigned int seed, bool random, double starting_range)
    {
        embeddedPoints.resize(data_amount);
        embeddedPointsPrev.resize(data_amount);
        embeddedPointsPrevPrev.resize(data_amount);

        std::mt19937 gen = [random, seed]() -> std::mt19937
        {
            if (random)
            {
                std::random_device rd;
                return std::mt19937(rd());
            }
            else
            {
                return std::mt19937(seed);
            }
        }();

        std::normal_distribution<double> dist(0.0, 0.01);

        for (int i = 0; i < data_amount; i++)
        {
            glm::dvec2 pos = glm::dvec2(dist(gen), dist(gen));

            int lab = labels[i];

#ifdef INDEX_TRACKER
            embeddedPoints[i] = TsnePoint2D(pos, glm::dvec2(0.0), lab, i, 0.0);
            embeddedPointsPrev[i] = TsnePoint2D(pos, glm::dvec2(0.0), lab, i, 0.0);
            embeddedPointsPrevPrev[i] = TsnePoint2D(pos, glm::dvec2(0.0), lab, i, 0.0);
#else
            embeddedPoints[i] = TsnePoint2D(pos, glm::dvec2(0.0), lab);
            embeddedPointsPrev[i] = TsnePoint2D(pos, glm::dvec2(0.0), lab);
            embeddedPointsPrevPrev[i] = TsnePoint2D(pos, glm::dvec2(0.0), lab);
#endif
        }
    }
};
