#pragma once

#include <FMM_tSNE/data_handling/data_entries/sparse_entry_COO_2D.h>
#include <FMM_tSNE/data_handling/data_entries/sparse_entry_CSC_2D.h>
#include <FMM_tSNE/data_handling/data_loading/MNIST_data.h>
#include <FMM_tSNE/data_handling/data_loading/sparse_data_loader.h>
#include <FMM_tSNE/data_handling/data_processing/data_processing.h>
#include <FMM_tSNE/opengl_interaction/scene.h>
#include <FMM_tSNE/timer/timer.h>
#include <FMM_tSNE/tsne/Q_total_cuda.h>
#include <FMM_tSNE/tsne/points/tsne_point_2D.h>
#include <FMM_tSNE/tsne/solvers/BH/solver_BH.h>
#include <FMM_tSNE/tsne/solvers/FMM/solver_FMM.h>
#include <FMM_tSNE/tsne/solvers/PM/solver_PM.h>
#include <FMM_tSNE/tsne/solvers/P_matrix/P_matrix.h>
#include <FMM_tSNE/tsne/solvers/SFMM/solver_SFMM.h>
#include <FMM_tSNE/tsne/solvers/naive/solver_naive.h>
#include <FMM_tSNE/tsne/solvers/naive_GPU/solver_naive_GPU.h>
#include <FMM_tSNE/tsne/solvers/solver.h>
#include <FMM_tSNE/tsne/solvers/solver_tree.h>
#include <GLFW/glfw3.h>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <glm/glm.hpp>
#include <limits>
#include <map>
#include <numbers>
#include <random>
#include <string>
#include <unsupported/Eigen/SparseExtra>
#include <utility>
#include <vector>

class TSNE
{
  public:
    double factor_ = 1.0;
    double QijTotal_ = 0.0;

    std::vector<TsnePoint2D> embeddedPoints;
    std::vector<TsnePoint2D> embeddedPointsPrev;
    std::vector<TsnePoint2D> embeddedPointsPrevPrev;

    glm::dvec2 minPos{std::numeric_limits<double>::max()};
    glm::dvec2 maxPos{std::numeric_limits<double>::lowest()};

    std::map<std::string, NBodySolver<TsnePoint2D>*> nBodySolvers;
    std::string nBodySelect{"naive"}; // default selector

    double learnRate{0.0};

    PMatrix p_matrix;

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

        p_matrix = PMatrix(DataProcessing::PCA(MNIST_data::load_MNIST_images_fused(data_amount), 50),
                           MNIST_data::load_MNIST_labels_fused(data_amount),
                           30);

        generateInitialPoints(data_amount, seed, false, 2.0);

        updateMinMaxPos();

        int max_children_per_node = 16;
        // nBodySolvers["naive"] = new NBodySolverNaive<TsnePoint2D>(&TSNEnaiveKernel);
        nBodySolvers["naive"] = new NBodySolverNaiveCUDA<TsnePoint2D>();
        // nBodySolvers["BH"] = new NBodySolverTree<TsnePoint2D, Policy_BH>(max_children_per_node, max_theta);
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
        nBodySolvers["PM"] = new NBodySolverPM<TsnePoint2D>(embeddedPoints, 4, cell_size, 4);
        nBodySolvers["PM"]->updateTree(embeddedPoints, minPos, maxPos);
#ifdef INDEX_TRACKER
        // nBodySolvers["FMM_MORTON"] = new NBodySolverFMM_MORTON<TsnePoint2D>(
        //     &TSNEFMM_MORTONNNKernel,
        //     &TSNEFMM_MORTONPNKernel,
        //     &TSNEFMM_MORTONNPKernel,
        //     &TSNEFMM_MORTONPPKernel,
        //     max_children_per_node,
        //     NBodySolverFMM_MORTON<TsnePoint2D>::getDepth(max_children_per_node, data_amount),
        //     max_theta);
        // nBodySolvers["FMM_MORTON"]->updateTree(embeddedPoints, minPos, maxPos);
        nBodySolvers["FMM_MORTON"] = new SolverFMM<TsnePoint2D>(max_children_per_node, max_theta);
        nBodySolvers["FMM_MORTON"]->updateTree(embeddedPoints, minPos, maxPos);
        nBodySolvers["FMM_SYM_MORTON"] = new SolverSFMM<TsnePoint2D>(max_children_per_node, max_theta);
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

        p_matrix = PMatrix(DataProcessing::PCA(MNIST_data::load_MNIST_images_fused(data_amount), 50),
                           MNIST_data::load_MNIST_labels_fused(data_amount),
                           30);

        generateInitialPoints(data_amount, seed, false, 2.0);

        updateMinMaxPos();

        setThetaForAll(min_theta, cell_size);
        // nBodySolvers["BH"]->updateTree(embeddedPoints, minPos, maxPos);
        // nBodySolvers["BHR"]->updateTree(embeddedPoints, minPos, maxPos);
        // nBodySolvers["BHMP"]->updateTree(embeddedPoints, minPos, maxPos);
        // nBodySolvers["BHRMP"]->updateTree(embeddedPoints, minPos, maxPos);
        // nBodySolvers["FMM"]->updateTree(embeddedPoints, minPos, maxPos);
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

        double accelerationRate = iteration_counter < 250 ? 0.5 : 0.8; // 250

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

        // std::cout << "set theta to: " << theta_result << std::endl;
        // std::cout << "set cell size to: " << cell_size << std::endl;

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
        QijTotal_ = 0.0;
        nBodySolvers[nBodySelect]->solveNbody(QijTotal_, embeddedPoints);
        QijTotal_ *= factor_;
        QijTotal_ = 4.0 / QijTotal_;

        for (int i = 0; i < embeddedPoints.size(); i++)
            embeddedPoints[i].derivative *= QijTotal_;

#ifdef INDEX_TRACKER
        for (size_t i = 0; i < embeddedPointsPrevPrev.size(); i++)
            embeddedPointsPrevPrev[embeddedPoints[i].ID] = embeddedPoints[i];
        embeddedPointsPrevPrev.swap(embeddedPoints);
#endif
    }

    void updateRepulsiveNoNorm()
    {
        QijTotal_ = 0.0;
        nBodySolvers[nBodySelect]->solveNbody(QijTotal_, embeddedPoints);
        QijTotal_ *= factor_;
        QijTotal_ = 4.0 / QijTotal_;

#ifdef INDEX_TRACKER
        for (size_t i = 0; i < embeddedPointsPrevPrev.size(); i++)
            embeddedPointsPrevPrev[embeddedPoints[i].ID] = embeddedPoints[i];
        embeddedPointsPrevPrev.swap(embeddedPoints);
#endif
    }

    void updateAttractive()
    {
        double exageration = iteration_counter < 250 ? 16.0 : 4.0; // the early exaggeration is 4.0f
        // double exageration = iteration_counter < 250 ? 4.0 : 4.0; // the early exaggeration is 4.0f

        for (int n = 0; n < embeddedPoints.size(); n++)
        {
            TsnePoint2D& pointR = embeddedPoints[n];

            glm::dvec2 dim{0.0f};

            for (unsigned int i = p_matrix.m_row_P[n]; i < p_matrix.m_row_P[n + 1]; i++)
            {
                unsigned int col = p_matrix.m_col_P[i];
                TsnePoint2D& pointC = embeddedPoints[col];

                glm::dvec2 diff = pointC.position - pointR.position;
                double d_ij = diff.x * diff.x + diff.y * diff.y;
                double q_ij = 1.0 / (1.0 + d_ij);

                dim += exageration * p_matrix.m_val_P[i] * q_ij * diff;
            }

            pointR.derivative += dim;
        }
    }

    static double costFunction(const std::vector<TsnePoint2D>& points_og, PMatrix& p_matrix)
    {
        std::vector<TsnePoint2D> points = points_og;

#ifdef INDEX_TRACKER
        for (size_t i = 0; i < points.size(); i++)
            points[points_og[i].ID] = points_og[i];
#endif

        QTotalCuda qTotalGpu;
        std::vector<double> qTotalPosBuffer;

        const int n = static_cast<int>(points.size());
        qTotalPosBuffer.resize(2 * static_cast<size_t>(n));
        for (int i = 0; i < n; i++)
        {
            qTotalPosBuffer[2 * i + 0] = points[i].position.x;
            qTotalPosBuffer[2 * i + 1] = points[i].position.y;
        }

        const double QijTotal = qTotalGpu.compute(qTotalPosBuffer.data(), n);

        double totalCost = 0.0;
        for (int row = 0; row < points_og.size(); row++)
        {
            const TsnePoint2D& row_point = points_og[row];

            glm::dvec2 dim{0.0f};

            for (unsigned int column = p_matrix.m_row_P[row]; column < p_matrix.m_row_P[row + 1]; column++)
            {
                const TsnePoint2D& col_point = points_og[p_matrix.m_col_P[column]];

                glm::dvec2 diff = col_point.position - row_point.position;
                double distance_squared = diff.x * diff.x + diff.y * diff.y;
                double Qij = (1.0 / (1.0 + distance_squared)) / QijTotal;

                double Pij = p_matrix.m_val_P[column];

                totalCost += Pij * std::log2(Pij / Qij);
            }
        }

        std::cout << "total cost: " << totalCost << std::endl;
        return totalCost;
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

            int lab = p_matrix.m_labels[i];

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
