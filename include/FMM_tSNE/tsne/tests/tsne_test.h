#pragma once

#include <FMM_tSNE/tsne/points/tsne_point_2D.h>
#include <FMM_tSNE/tsne/tsne.h>
#include <GLFW/glfw3.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtx/component_wise.hpp>
#include <glm/gtx/string_cast.hpp>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <print>
#include <ranges>
#include <string>
#include <thread>
#include <unsupported/Eigen/SparseExtra>
#include <utility>
#include <vector>

class TsneTest
{
  public:
    TsneTest() {}

    void errorTimestepTSNE(std::string dataset_type,
                           int data_size,
                           float perplexity_value,
                           int iteration_amount,
                           double theta,
                           double cell_size,
                           unsigned int seed)
    {
        TSNE tsne(theta, theta, cell_size, dataset_type, data_size, perplexity_value, seed);

        std::vector<TsnePoint2D> naive_solution_rep(data_size);
        std::vector<TsnePoint2D> fast_solution_rep(data_size);

        std::vector<TsnePoint2D> naive_solution(data_size);

        // set graph size
        std::vector<std::string> candidates;
        candidates.reserve(tsne.nBodySolvers.size());
        std::ranges::transform(
            tsne.nBodySolvers, std::back_inserter(candidates), [](const auto& pair) { return pair.first; });

        std::map<std::string, std::vector<double>> errors;
        std::map<std::string, std::vector<int>> iteration_counts;

        for (std::string candidate : candidates)
        {
            errors.insert({candidate, std::vector<double>(iteration_amount, 0.0)});
            iteration_counts.insert({candidate, std::vector<int>(iteration_amount, 0)});
        }

        // std::string indexCheck = "noindex";

        // std::string full = "full";
        // std::string compare = "comp";
        // std::filesystem::path pre_computed_path_full_naive_solution = std::filesystem::current_path() /
        // std::string("precomputed_states") / dataset_type / std::to_string(data_size) /
        // std::to_string(static_cast<unsigned int>(perplexity_value)) / std::to_string(seed) / indexCheck / full;
        // std::filesystem::path pre_computed_path_naive_solution = std::filesystem::current_path() /
        // std::string("precomputed_states") / dataset_type / std::to_string(data_size) /
        // std::to_string(static_cast<unsigned int>(perplexity_value)) / std::to_string(seed) / indexCheck / compare;

        // if (!std::filesystem::exists(pre_computed_path_full_naive_solution))
        //     std::filesystem::create_directories(pre_computed_path_full_naive_solution);

        // if (!std::filesystem::exists(pre_computed_path_naive_solution))
        //     std::filesystem::create_directories(pre_computed_path_naive_solution);

        // find error at every time step
        for (int t = 0; t < iteration_amount; t++)
        {
            std::cout << "find error at iteration: " << t << std::endl;
            tsne.updateMinMaxPos();

            // correct solution up to machine precision
            tsne.resetDeriv();
            tsne.iteration_counter = t;
            tsne.nBodySelect = "naive";
            tsne.updateDerivative();
            naive_solution = tsne.embeddedPoints;
            // storePrecomputed(naive_solution, pre_computed_path_full_naive_solution / std::to_string(t));

            tsne.resetDeriv();
            tsne.iteration_counter = t;
            tsne.nBodySelect = "naive";
            tsne.resetDeriv();
            tsne.updateRepulsive();
            // tsne.updateRepulsiveNoNorm();
            naive_solution_rep = tsne.embeddedPoints;
            // storePrecomputed(naiveSolution, pre_computed_path_naive_solution / std::to_string(t));

            for (std::string candidate : candidates)
            {
                // calculate the result of every approximation technique and find the error by comparing to naive
                tsne.resetDeriv();
                tsne.iteration_counter = t;
                tsne.nBodySelect = candidate;
                tsne.nBodySolvers[candidate]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                // tsne.updateDerivative();
                tsne.resetDeriv();
                tsne.updateRepulsive();
                // tsne.updateRepulsiveNoNorm();
                fast_solution_rep = tsne.embeddedPoints;
                // fastSolutionIndexTracker = tsne.indexTracker;
                // errors[candidate][t] = getNMAE(naive_solution_rep, fast_solution_rep);
                errors[candidate][t] = getNMAE(naive_solution_rep, fast_solution_rep);
                iteration_counts[candidate][t] = t;
            }

            // update points with naive solution
            tsne.resetDeriv();
            tsne.iteration_counter = t;
            tsne.nBodySelect = "naive";
            tsne.embeddedPoints = naive_solution;
            tsne.updatePoints();
        }

        // write results to csv files
        std::filesystem::path projectFolder = std::filesystem::current_path() / "graphCSV" / dataset_type /
                                              std::to_string(data_size) /
                                              ("perp" + std::to_string(static_cast<int>(perplexity_value)));

        for (std::string candidate : candidates)
        {
            std::filesystem::path methodPath =
                projectFolder / ("tsneErrorTimestep" + candidate + "_point" + std::to_string(data_size) + "_perp" +
                                 std::to_string(static_cast<int>(perplexity_value)) + "_theta" + fltToStr(theta) +
                                 "_dataset" + dataset_type + ".csv");
            writeToFile(iteration_counts[candidate], errors[candidate], methodPath);
        }
    }

    void calculationtimeThetaTSNE(std::string dataset_type,
                                  int data_size,
                                  float perplexity_value,
                                  int iteration_amount,
                                  std::vector<double> thetas,
                                  std::vector<double> cell_sizes,
                                  unsigned int seed)
    {
        int theta_diversity_amount = thetas.size();

        TSNE tsne(thetas[0], thetas[0], cell_sizes[0], dataset_type, data_size, perplexity_value, seed);

        std::vector<std::string> candidates;
        candidates.reserve(tsne.nBodySolvers.size());
        std::ranges::transform(
            tsne.nBodySolvers, std::back_inserter(candidates), [](const auto& pair) { return pair.first; });

        std::map<std::string, std::vector<double>> calc_times;
        std::map<std::string, std::vector<double>> thetas_arrays;
        for (std::string candidate : candidates)
        {
            calc_times.insert({candidate, std::vector<double>(theta_diversity_amount, 0.0)});
            thetas_arrays.insert({candidate, std::vector<double>(theta_diversity_amount, 0.0)});
        }

        for (int t = 0; t < theta_diversity_amount; t++)
        {
            double chosen_theta = thetas[t];
            double chosen_cell_size = cell_sizes[t];

            std::print("about to reset tsne with following:\n{}\n{}\n{}\n{}\n{}\n",
                       chosen_theta,
                       chosen_cell_size,
                       dataset_type,
                       data_size,
                       perplexity_value);
            tsne.resetTsne(
                chosen_theta, chosen_theta, chosen_cell_size, dataset_type, data_size, perplexity_value, seed);
            std::print("done resetting\n");

            for (std::string candidate : candidates)
            {
                thetas_arrays[candidate][t] = chosen_theta;
            }

            std::vector<TsnePoint2D> naive_solution(data_size);

            // #ifdef INDEX_TRACKER
            // std::string indexCheck = "index";
            // #else
            // std::string indexCheck = "noindex";
            // #endif
            // std::string full = "full";
            // std::filesystem::path pre_computed_path_full_naive_solution = std::filesystem::current_path() /
            // std::string("precomputed_states") / dataset_type / std::to_string(data_size) /
            // std::to_string(static_cast<unsigned int>(perplexity_value)) / std::to_string(seed) / indexCheck / full;

            // if (!std::filesystem::exists(pre_computed_path_full_naive_solution))
            //     std::filesystem::create_directories(pre_computed_path_full_naive_solution);

            double timeBefore = 0.0;
            for (int j = 0; j < iteration_amount; j++)
            {
                // std::vector<TsnePoint2D>& naiveSolution = naiveSolutions[j];
                // std::vector<int>& naiveSolutionIndexTracker = naiveSolutionIndexTrackers[j];

                std::cout << "find calc time at iteration: " << j << ", for theta: " << chosen_theta << std::endl;

                tsne.updateMinMaxPos();

                // correct solution up to machine precision
                // if (std::filesystem::exists(pre_computed_path_full_naive_solution / std::to_string(j)))
                //{
                //     loadPrecomputed(naive_solution, pre_computed_path_full_naive_solution / std::to_string(j));
                // }
                // else
                //{
                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "naive";
                tsne.updateDerivative();
                naive_solution = tsne.embeddedPoints;
                // storePrecomputed(naive_solution, pre_computed_path_full_naive_solution / std::to_string(j));
                //}

                for (std::string candidate : candidates)
                {
                    tsne.resetDeriv();
                    tsne.iteration_counter = j;
                    tsne.nBodySelect = candidate;
                    timeBefore = glfwGetTime();
                    tsne.nBodySolvers[candidate]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                    // tsne.updateDerivative();
                    tsne.resetDeriv();
                    tsne.updateRepulsive();
                    calc_times[candidate][t] += glfwGetTime() - timeBefore;
                }

                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "naive";
                tsne.embeddedPoints = naive_solution;
                tsne.updatePoints();
            }

            for (std::string candidate : candidates)
            {
                calc_times[candidate][t] /= static_cast<double>(iteration_amount);
            }
        }

        // write results to csv files
        std::filesystem::path projectFolder = std::filesystem::current_path() / "graphCSV" / dataset_type /
                                              std::to_string(data_size) /
                                              ("perp" + std::to_string(static_cast<int>(perplexity_value)));

        for (std::string candidate : candidates)
        {
            std::filesystem::path methodPath =
                projectFolder /
                ("tsneCalculationtimeTheta" + candidate + "_point" + std::to_string(data_size) + "_perp" +
                 std::to_string(static_cast<int>(perplexity_value)) + "_dataset" + dataset_type + ".csv");
            writeToFile(thetas_arrays[candidate], calc_times[candidate], methodPath);
        }
    }

    void errorThetaTSNE(std::string dataset_type,
                        int data_size,
                        float perplexity_value,
                        int iteration_amount,
                        std::vector<float> thetas,
                        std::vector<double> cell_sizes,
                        unsigned int seed)
    {
        TSNE tsne(thetas[0], thetas[0], cell_sizes[0], dataset_type, data_size, perplexity_value, seed);

        int theta_diversity_amount = thetas.size();

        std::vector<std::string> candidates;
        candidates.reserve(tsne.nBodySolvers.size());
        std::ranges::transform(
            tsne.nBodySolvers, std::back_inserter(candidates), [](const auto& pair) { return pair.first; });

        std::map<std::string, std::vector<double>> errors;
        std::map<std::string, std::vector<double>> thetas_arrays;

        for (std::string candidate : candidates)
        {
            errors.insert({candidate, std::vector<double>(theta_diversity_amount, 0.0)});
            thetas_arrays.insert({candidate, std::vector<double>(theta_diversity_amount, 0.0)});
        }

        for (int t = 0; t < theta_diversity_amount; t++)
        {
            double chosen_theta = thetas[t];
            double chosen_cell_size = cell_sizes[t];

            tsne.resetTsne(
                chosen_theta, chosen_theta, chosen_cell_size, dataset_type, data_size, perplexity_value, seed);

            for (std::string candidate : candidates)
            {
                thetas_arrays[candidate][t] = chosen_theta;
            }

            std::vector<TsnePoint2D> naive_solution_rep(data_size);
            std::vector<TsnePoint2D> fast_solution_rep(data_size);

            std::vector<TsnePoint2D> naive_solution(data_size);

            // #ifdef INDEX_TRACKER
            // std::string indexCheck = "index";
            // #else
            // std::string indexCheck = "noindex";
            // #endif
            // std::string full = "full";
            // std::string compare = "comp";
            // std::filesystem::path pre_computed_path_full_naive_solution = std::filesystem::current_path() /
            // std::string("precomputed_states") / dataset_type / std::to_string(data_size) /
            // std::to_string(static_cast<unsigned int>(perplexity_value)) / std::to_string(seed) / indexCheck / full;
            // std::filesystem::path pre_computed_path_naive_solution = std::filesystem::current_path() /
            // std::string("precomputed_states") / dataset_type / std::to_string(data_size) /
            // std::to_string(static_cast<unsigned int>(perplexity_value)) / std::to_string(seed) / indexCheck /
            // compare;

            // if (!std::filesystem::exists(pre_computed_path_full_naive_solution))
            //     std::filesystem::create_directories(pre_computed_path_full_naive_solution);

            // if (!std::filesystem::exists(pre_computed_path_naive_solution))
            //     std::filesystem::create_directories(pre_computed_path_naive_solution);

            for (int j = 0; j < iteration_amount; j++)
            {
                std::cout << "find error at iteration: " << j << ", for theta: " << chosen_theta << std::endl;

                tsne.updateMinMaxPos();

                // correct solution up to machine precision
                // if (std::filesystem::exists(pre_computed_path_full_naive_solution / std::to_string(j)))
                //{
                //    loadPrecomputed(naive_solution, pre_computed_path_full_naive_solution / std::to_string(j));
                //}
                // else
                //{
                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "naive";
                tsne.updateDerivative();
                naive_solution = tsne.embeddedPoints;
                // storePrecomputed(naive_solution, pre_computed_path_full_naive_solution / std::to_string(j));
                //}

                // if (std::filesystem::exists(pre_computed_path_naive_solution / std::to_string(j)))
                //{
                //     loadPrecomputed(naive_solution_rep, pre_computed_path_naive_solution / std::to_string(j));
                // }
                // else
                //{
                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "naive";
                tsne.resetDeriv();
                tsne.updateRepulsive();
                naive_solution_rep = tsne.embeddedPoints;
                // storePrecomputed(naive_solution_rep, pre_computed_path_naive_solution / std::to_string(j));
                //}

                for (std::string candidate : candidates)
                {
                    tsne.resetDeriv();
                    tsne.iteration_counter = j;
                    tsne.nBodySelect = candidate;
                    tsne.nBodySolvers[candidate]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                    tsne.resetDeriv();
                    tsne.updateRepulsive();
                    fast_solution_rep = tsne.embeddedPoints;
                    errors[candidate][t] += getNMAE(naive_solution_rep, fast_solution_rep);
                }

                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "naive";
                tsne.embeddedPoints = naive_solution;
                tsne.updatePoints();
            }

            for (std::string candidate : candidates)
            {
                errors[candidate][t] /= static_cast<double>(iteration_amount);
            }
        }

        // write results to csv files
        std::filesystem::path projectFolder = std::filesystem::current_path() / "graphCSV" / dataset_type /
                                              std::to_string(data_size) /
                                              ("perp" + std::to_string(static_cast<int>(perplexity_value)));

        for (std::string candidate : candidates)
        {
            std::filesystem::path methodPath =
                projectFolder /
                ("tsneErrorTheta" + candidate + "_point" + std::to_string(data_size) + "_perp" +
                 std::to_string(static_cast<int>(perplexity_value)) + "_dataset" + dataset_type + ".csv");
            writeToFile(thetas_arrays[candidate], errors[candidate], methodPath);
        }
    }

    void calculationtimeErrorTSNE(std::string dataset_type,
                                  int data_size,
                                  float perplexity_value,
                                  int iteration_amount,
                                  std::vector<double> thetas,
                                  std::vector<double> cell_sizes,
                                  unsigned int seed)
    {
        TSNE tsne(thetas[0], 3.0, cell_sizes[0], dataset_type, data_size, perplexity_value, seed);

        int theta_diversity_amount = thetas.size();

        std::vector<std::string> candidates;
        candidates.reserve(tsne.nBodySolvers.size());
        std::ranges::transform(
            tsne.nBodySolvers, std::back_inserter(candidates), [](const auto& pair) { return pair.first; });

        std::map<std::string, std::vector<double>> times;
        std::map<std::string, std::vector<double>> errors;
        std::map<std::string, std::vector<double>> thetas_arrays;

        for (std::string candidate : candidates)
        {
            times.insert({candidate, std::vector<double>(theta_diversity_amount, 0.0)});
            errors.insert({candidate, std::vector<double>(theta_diversity_amount, 0.0)});
            thetas_arrays.insert({candidate, std::vector<double>(theta_diversity_amount, 0.0)});
        }

        for (int t = 0; t < theta_diversity_amount; t++)
        {
            double chosen_theta = thetas[t];
            double chosen_cell_size = cell_sizes[t];

            tsne.resetTsne(chosen_theta, 3.0, chosen_cell_size, dataset_type, data_size, perplexity_value, seed);

            std::vector<TsnePoint2D> naive_solution_rep(data_size);
            std::vector<TsnePoint2D> fast_solution_rep(data_size);

            std::vector<TsnePoint2D> naive_solution(data_size);

            // #ifdef INDEX_TRACKER
            // std::string indexCheck = "index";
            // #else
            // std::string indexCheck = "noindex";
            // #endif
            // std::string full = "full";
            // std::string compare = "comp";
            // std::filesystem::path pre_computed_path_full_naive_solution = std::filesystem::current_path() /
            // std::string("precomputed_states") / dataset_type / std::to_string(data_size) /
            // std::to_string(static_cast<unsigned int>(perplexity_value)) / std::to_string(seed) / indexCheck / full;
            // std::filesystem::path pre_computed_path_naive_solution = std::filesystem::current_path() /
            // std::string("precomputed_states") / dataset_type / std::to_string(data_size) /
            // std::to_string(static_cast<unsigned int>(perplexity_value)) / std::to_string(seed) / indexCheck /
            // compare;

            // if (!std::filesystem::exists(pre_computed_path_full_naive_solution))
            //     std::filesystem::create_directories(pre_computed_path_full_naive_solution);

            // if (!std::filesystem::exists(pre_computed_path_naive_solution))
            //     std::filesystem::create_directories(pre_computed_path_naive_solution);

            double timeBefore = 0.0f;
            for (int j = 0; j < iteration_amount; j++)
            {
                std::cout << "find calc time and error at iteration: " << j << ", for theta: " << chosen_theta
                          << std::endl;

                tsne.updateMinMaxPos();

                // correct solution up to machine precision
                // if (std::filesystem::exists(pre_computed_path_full_naive_solution / std::to_string(j)))
                //{
                //    loadPrecomputed(naive_solution, pre_computed_path_full_naive_solution / std::to_string(j));
                //}
                // else
                //{
                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "naive";
                tsne.updateDerivative();
                naive_solution = tsne.embeddedPoints;
                // storePrecomputed(naive_solution, pre_computed_path_full_naive_solution / std::to_string(j));
                //}

                // if (std::filesystem::exists(pre_computed_path_naive_solution / std::to_string(j)))
                //{
                //     loadPrecomputed(naive_solution_rep, pre_computed_path_naive_solution / std::to_string(j));
                // }
                // else
                //{
                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "naive";
                tsne.resetDeriv();
                tsne.updateRepulsive();
                naive_solution_rep = tsne.embeddedPoints;
                // storePrecomputed(naive_solution_rep, pre_computed_path_naive_solution / std::to_string(j));
                //}

                for (std::string candidate : candidates)
                {
                    tsne.resetDeriv();
                    tsne.iteration_counter = j;
                    tsne.nBodySelect = candidate;
                    timeBefore = glfwGetTime();
                    tsne.nBodySolvers[candidate]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                    // tsne.updateDerivative();
                    tsne.resetDeriv();
                    tsne.updateRepulsive();
                    times[candidate][t] += glfwGetTime() - timeBefore;
                    fast_solution_rep = tsne.embeddedPoints;
                    // fastSolutionIndexTracker = tsne.indexTracker;
                    errors[candidate][t] += getNMAE(naive_solution_rep, fast_solution_rep);
                }

                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "naive";
                tsne.embeddedPoints = naive_solution;
                tsne.updatePoints();
            }

            for (std::string candidate : candidates)
            {
                errors[candidate][t] /= iteration_amount;
                times[candidate][t] /= iteration_amount;
                thetas_arrays[candidate][t] = chosen_theta;
            }
        }

        // write results to csv files
        std::filesystem::path projectFolder = std::filesystem::current_path() / "graphCSV" / dataset_type /
                                              std::to_string(data_size) /
                                              ("perp" + std::to_string(static_cast<int>(perplexity_value)));

        for (std::string candidate : candidates)
        {
            std::filesystem::path methodPath =
                projectFolder /
                ("tsneCalculationtimeError" + candidate + "_point" + std::to_string(data_size) + "_perp" +
                 std::to_string(static_cast<int>(perplexity_value)) + "_dataset" + dataset_type + ".csv");
            writeToFile3(errors[candidate], times[candidate], thetas_arrays[candidate], methodPath);
        }
    }

    void costTimestepTSNE(std::string dataset_type,
                          int data_size,
                          float perplexity_value,
                          int iteration_amount,
                          double min_theta,
                          double max_theta,
                          double cell_size,
                          unsigned int seed,
                          std::string method)
    {
        TSNE tsne(min_theta, max_theta, cell_size, dataset_type, data_size, perplexity_value, seed);

        // set graph size
        std::vector<double> costs(iteration_amount, 0.0);

        std::vector<int> times(iteration_amount, 0);

        // find error at every time step
        for (int t = 0; t < iteration_amount; t++)
        {
            std::cout << "find error at iteration: " << t << std::endl;

            tsne.updateMinMaxPos();

            // calculate the result of every approximation technique and find the error by comparing to naive
            tsne.resetDeriv();
            tsne.iteration_counter = t;
            tsne.nBodySelect = method;
            tsne.nBodySolvers[method]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
            tsne.updateDerivative();
            if (t % 10 == 0)
            {
                costs[t] = costFunction(tsne.embeddedPoints, tsne.Pmatrix);

                if (t > 0)
                {
                    double prevCost = costs[t - 10];
                    double currCost = costs[t];

                    for (int i = 1; i < 10; ++i)
                    {
                        double alpha = static_cast<double>(i) / 10.0;
                        costs[t - 10 + i] = prevCost + alpha * (currCost - prevCost);
                    }
                }
            }
            times[t] = t;

            tsne.updatePoints();
        }

        // write results to csv files
        std::filesystem::path projectFolder = std::filesystem::current_path() / "graphCSV" / dataset_type /
                                              std::to_string(data_size) /
                                              ("perp" + std::to_string(static_cast<int>(perplexity_value)));

        std::filesystem::path methodPath;
        methodPath =
            projectFolder / ("costTimestep" + std::string(method) + "_point" + std::to_string(data_size) + "_perp" +
                             std::to_string(static_cast<int>(perplexity_value)) + "_minTheta" + fltToStr(min_theta) +
                             "_maxTheta" + fltToStr(max_theta) + "_dataset" + dataset_type + ".csv");
        writeToFile(times, costs, methodPath);
    }

    void calculationtimeCostTSNE(std::string dataset_type,
                                 int data_size,
                                 float perplexity_value,
                                 int iteration_amount,
                                 std::vector<double> thetas,
                                 std::vector<double> cell_sizes,
                                 unsigned int seed,
                                 std::string method)
    {
        int theta_diversity_amount = thetas.size();
        std::vector<double> calculationtime(theta_diversity_amount, 0.0f);

        std::vector<double> finalCost(theta_diversity_amount, 0.0f);

        // std::vector<float> thetas(theta_diversity_amount, 0.0f);

        for (int t = 0; t < theta_diversity_amount; t++)
        {
            double chosenTheta = thetas[t];
            double chosenCellSize = cell_sizes[t];

            TSNE tsne(chosenTheta, 3.0, chosenCellSize, dataset_type, data_size, perplexity_value, seed);

            double timeBefore = 0.0f;
            for (int j = 0; j < iteration_amount; j++)
            {
                std::cout << "find calc time and error at iteration: " << j << ", for theta: " << chosenTheta
                          << std::endl;

                tsne.updateMinMaxPos();

                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = method;
                timeBefore = glfwGetTime();
                tsne.nBodySolvers[method]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                tsne.updateDerivative();
                calculationtime[t] += glfwGetTime() - timeBefore;

                tsne.updatePoints();
            }

            finalCost[t] = costFunction(tsne.embeddedPoints, tsne.Pmatrix);

            calculationtime[t] /= iteration_amount;
        }

        // write results to csv files
        std::filesystem::path projectFolder = std::filesystem::current_path() / "graphCSV" / dataset_type /
                                              std::to_string(data_size) /
                                              ("perp" + std::to_string(static_cast<int>(perplexity_value)));

        std::filesystem::path methodPath;
        methodPath = projectFolder /
                     ("tsneCalculationtimeCost" + std::string(method) + "_point" + std::to_string(data_size) + "_perp" +
                      std::to_string(static_cast<int>(perplexity_value)) + "_dataset" + dataset_type + ".csv");
        writeToFile3(finalCost, calculationtime, thetas, methodPath);
    }

    void calculationtimeError_MAXpoints_TSNE(std::string dataset_type,
                                             int data_size,
                                             float perplexity_value,
                                             int iteration_amount,
                                             std::vector<double> thetas,
                                             std::vector<int> max_points,
                                             unsigned int seed,
                                             std::string method)
    {
        for (size_t m = 0; m < max_points.size(); m++)
        {
            std::cout << "find calc time and error for max points: " << max_points[m] << std::endl;

            int theta_diversity_amount = thetas.size();

            std::vector<double> calculationtime(theta_diversity_amount, 0.0f);

            std::vector<double> error(theta_diversity_amount, 0.0f);

            std::vector<double> theta_tracker(theta_diversity_amount, 0.0f);

            for (int t = 0; t < theta_diversity_amount; t++)
            {
                double chosenTheta = thetas[t];

                TSNE tsne(chosenTheta, chosenTheta, 1.0, dataset_type, data_size, perplexity_value, seed);

                double max_point = max_points[m];
                tsne.nBodySolvers[method]->maxChildren = max_point;

                std::vector<TsnePoint2D> naiveSolution(data_size);
                std::vector<TsnePoint2D> fastSolution(data_size);

                std::vector<TsnePoint2D> full_naive_solution(data_size);

#ifdef INDEX_TRACKER
                std::string indexCheck = "index";
#else
                std::string indexCheck = "noindex";
#endif
                std::string full = "full";
                std::string compare = "comp";
                std::filesystem::path pre_computed_path_full_naive_solution =
                    std::filesystem::current_path() / std::string("precomputed_states") / dataset_type /
                    std::to_string(data_size) / std::to_string(static_cast<unsigned int>(perplexity_value)) /
                    std::to_string(seed) / indexCheck / full;
                std::filesystem::path pre_computed_path_naive_solution =
                    std::filesystem::current_path() / std::string("precomputed_states") / dataset_type /
                    std::to_string(data_size) / std::to_string(static_cast<unsigned int>(perplexity_value)) /
                    std::to_string(seed) / indexCheck / compare;

                if (!std::filesystem::exists(pre_computed_path_full_naive_solution))
                    std::filesystem::create_directories(pre_computed_path_full_naive_solution);

                if (!std::filesystem::exists(pre_computed_path_naive_solution))
                    std::filesystem::create_directories(pre_computed_path_naive_solution);

                int measure_amount = 0;
                double timeBefore = 0.0f;
                for (int j = 0; j < iteration_amount; j++)
                {
                    std::cout << "find calc time and error at iteration: " << j << ", for theta: " << chosenTheta
                              << std::endl;

                    tsne.updateMinMaxPos();

                    // correct solution up to machine precision
                    if (std::filesystem::exists(pre_computed_path_full_naive_solution / std::to_string(j)))
                    {
                        loadPrecomputed(full_naive_solution, pre_computed_path_full_naive_solution / std::to_string(j));
                    }
                    else
                    {
                        tsne.resetDeriv();
                        tsne.iteration_counter = j;
                        tsne.nBodySelect = "naive";
                        tsne.updateDerivative();
                        full_naive_solution = tsne.embeddedPoints;
                        storePrecomputed(full_naive_solution,
                                         pre_computed_path_full_naive_solution / std::to_string(j));
                    }

                    if (j % 20 == 0)
                    {
                        measure_amount++;

                        if (std::filesystem::exists(pre_computed_path_naive_solution / std::to_string(j)))
                        {
                            loadPrecomputed(naiveSolution, pre_computed_path_naive_solution / std::to_string(j));
                        }
                        else
                        {
                            tsne.resetDeriv();
                            tsne.iteration_counter = j;
                            tsne.nBodySelect = "naive";
                            tsne.resetDeriv();
                            tsne.updateRepulsive();
                            naiveSolution = tsne.embeddedPoints;
                            storePrecomputed(naiveSolution, pre_computed_path_naive_solution / std::to_string(j));
                        }

                        tsne.resetDeriv();
                        tsne.iteration_counter = j;
                        tsne.nBodySelect = method;
                        timeBefore = glfwGetTime();
                        tsne.nBodySolvers[method]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                        // tsne.updateDerivative();
                        tsne.resetDeriv();
                        tsne.updateRepulsive();
                        calculationtime[t] += glfwGetTime() - timeBefore;
                        fastSolution = tsne.embeddedPoints;
                        error[t] += getNMAE(naiveSolution, fastSolution);
                    }

                    tsne.resetDeriv();
                    tsne.iteration_counter = j;
                    tsne.nBodySelect = "naive";
                    tsne.embeddedPoints = full_naive_solution;
                    tsne.updatePoints();
                }

                error[t] /= measure_amount;

                calculationtime[t] /= measure_amount;

                theta_tracker[t] = chosenTheta;
            }

            // write results to csv files
            std::filesystem::path projectFolder = std::filesystem::current_path() / "graphCSV" / dataset_type /
                                                  std::to_string(data_size) /
                                                  ("perp" + std::to_string(static_cast<int>(perplexity_value)));

            std::filesystem::path methodPath;

            methodPath = projectFolder /
                         ("tsneCalculationtimeErrorMAXPOINTS_" + std::to_string(max_points[m]) + method + "_point" +
                          std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) +
                          "_dataset" + dataset_type + ".csv");
            writeToFile3(error, calculationtime, theta_tracker, methodPath);
        }
    }

    void time_accuracy_test(std::string dataset_type,
                            int data_size,
                            float perplexity_value,
                            int iteration_amount,
                            int repeat_count,
                            double theta,
                            double cell_size,
                            unsigned int seed)
    {
        TSNE tsne(theta, theta, cell_size, dataset_type, data_size, perplexity_value, seed);

        std::vector<TsnePoint2D> naive_solution_rep(data_size);
        std::vector<TsnePoint2D> fast_solution_rep(data_size);

        std::vector<TsnePoint2D> naive_solution(data_size);
        std::vector<TsnePoint2D> fast_solution(data_size);

        double error{};

        for (int t = 0; t < iteration_amount; t++)
        {
            tsne.updateMinMaxPos();

            // correct solution up to machine precision
            tsne.resetDeriv();
            tsne.iteration_counter = t;
            tsne.nBodySelect = "naive";
            tsne.updateDerivative();
            naive_solution = tsne.embeddedPoints;

            tsne.resetDeriv();
            tsne.iteration_counter = t;
            tsne.nBodySelect = "naive";
            tsne.resetDeriv();
            tsne.updateRepulsive();
            naive_solution_rep = tsne.embeddedPoints;

            tsne.resetDeriv();
            tsne.iteration_counter = t;
            tsne.nBodySelect = "FMMfast_SYM_MORTON";
            tsne.nBodySolvers["FMMfast_SYM_MORTON"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
            // tsne.updateDerivative();
            tsne.resetDeriv();
            tsne.updateRepulsive();
            fast_solution_rep = tsne.embeddedPoints;
            error += getNMAE(naive_solution_rep, fast_solution_rep);

            // update points with naive solution
            tsne.resetDeriv();
            tsne.iteration_counter = t;
            tsne.nBodySelect = "naive";
            tsne.embeddedPoints = naive_solution;
            tsne.updatePoints();
        }
        std::cout << "error: " << error / iteration_amount << "\n";

        std::chrono::microseconds total_time{};

        for (int r{}; r < repeat_count; ++r)
        {
            tsne.resetTsne(theta, theta, cell_size, dataset_type, data_size, perplexity_value, seed);
            for (int t = 0; t < iteration_amount; t++)
            {
                tsne.updateMinMaxPos();

                tsne.resetDeriv();
                tsne.iteration_counter = t;
                tsne.nBodySelect = "FMMfast_SYM_MORTON";
                Timer timer{};
                tsne.nBodySolvers["FMMfast_SYM_MORTON"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                tsne.updateDerivative();
                total_time += timer.elapsed_time();
                fast_solution = tsne.embeddedPoints;

                tsne.resetDeriv();
                tsne.iteration_counter = t;
                tsne.nBodySelect = "FMMfast_SYM_MORTON";
                tsne.embeddedPoints = fast_solution;
                tsne.updatePoints();
            }
        }

        total_time /= repeat_count;

        std::cout << "time: " << total_time << "\n";
    }

    void errorTimestepTSNE_Zaccum(std::string dataset_type,
                                  int data_size,
                                  float perplexity_value,
                                  int iteration_amount,
                                  double theta,
                                  double cell_size,
                                  unsigned int seed)
    {
        TSNE tsne(theta, theta, cell_size, dataset_type, data_size, perplexity_value, seed);

        std::vector<TsnePoint2D> naive_solution_rep(data_size);
        std::vector<TsnePoint2D> fast_solution_rep(data_size);

        std::vector<TsnePoint2D> naive_solution(data_size);

        std::vector<double> factors;
        for (double offset{0.0}; offset < 0.02; offset += 0.004)
        {
            factors.push_back(1.015 + offset);
            factors.push_back(1.015 - offset);
        }
        std::sort(factors.begin(), factors.end());
        std::vector<double> errors(factors.size(), 0.0);

        for (int factor{}; factor < factors.size(); ++factor)
        {
            tsne.resetTsne(theta, theta, cell_size, dataset_type, data_size, perplexity_value, seed);

            // find error at every time step
            for (int t = 0; t < iteration_amount; t++)
            {
                tsne.factor_ = 1.0;
                std::cout << "find error at iteration: " << t << std::endl;
                tsne.updateMinMaxPos();

                // correct solution up to machine precision
                tsne.resetDeriv();
                tsne.iteration_counter = t;
                tsne.nBodySelect = "naive";
                tsne.updateDerivative();
                naive_solution = tsne.embeddedPoints;
                // storePrecomputed(naive_solution, pre_computed_path_full_naive_solution / std::to_string(t));

                tsne.resetDeriv();
                tsne.iteration_counter = t;
                tsne.nBodySelect = "naive";
                tsne.resetDeriv();
                tsne.updateRepulsive();
                naive_solution_rep = tsne.embeddedPoints;
                // storePrecomputed(naiveSolution, pre_computed_path_naive_solution / std::to_string(t));

                {
                    tsne.factor_ = factors[factor];
                    // calculate the result of every approximation technique and find the error by comparing to naive
                    tsne.resetDeriv();
                    tsne.iteration_counter = t;
                    tsne.nBodySelect = "FMMfast_SYM_MORTON";
                    tsne.nBodySolvers["FMMfast_SYM_MORTON"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                    // tsne.updateDerivative();
                    tsne.resetDeriv();
                    tsne.updateRepulsive();
                    fast_solution_rep = tsne.embeddedPoints;
                    // fastSolutionIndexTracker = tsne.indexTracker;
                    errors[factor] += getNMAE(naive_solution_rep, fast_solution_rep);
                }

                tsne.factor_ = 1.0;
                // update points with naive solution
                tsne.resetDeriv();
                tsne.iteration_counter = t;
                tsne.nBodySelect = "naive";
                tsne.embeddedPoints = naive_solution;
                tsne.updatePoints();
            }
            errors[factor] /= iteration_amount;
        }

        // write results to csv files
        std::filesystem::path projectFolder = std::filesystem::current_path() / "graphCSV" / dataset_type /
                                              std::to_string(data_size) /
                                              ("perp" + std::to_string(static_cast<int>(perplexity_value)));

        std::filesystem::path methodPath =
            projectFolder / (std::string("tsneErrorTimestepZaccum") + "FMMfast_SYM_MORTON" + "_point" +
                             std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) +
                             "_theta" + fltToStr(theta) + "_dataset" + dataset_type + ".csv");
        writeToFile(factors, errors, methodPath);
    }

  private:
    // mean relative error or maximum relative error
    double getNMAE(const std::vector<TsnePoint2D>& points_naive, const std::vector<TsnePoint2D>& points_approx)
    {
        double MRE = 0.0;
        double norm = 0.0;

        for (int i = 0; i < points_naive.size(); i++)
        {
            TsnePoint2D pointNaive = points_naive[i];
            TsnePoint2D pointApprox = points_approx[i];

            MRE += static_cast<double>(glm::length(pointNaive.derivative - pointApprox.derivative)) /
                   static_cast<double>(glm::length(pointNaive.derivative));

            // MRE = std::max(MRE, static_cast<double>(glm::length(pointNaive.derivative - pointApprox.derivative)) /
            // static_cast<double>(glm::length(pointNaive.derivative)));
        }

        MRE = MRE / points_naive.size();

        return MRE;
    }

    double costFunction(const std::vector<TsnePoint2D>& points, Eigen::SparseMatrix<double>& Pmatrix)
    {
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
                            localTotals[t] += static_cast<double>(1.0 / (1.0 + distance_squared));
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
                    double Qij = static_cast<double>(1.0 / (1.0 + distance_squared)) / QijTotal;

                    double Pij = it.value();

                    totalCost += Pij * std::log2(Pij / Qij);
                }
            }
        }

        // std::cout << "total cost: " << totalCost << std::endl;
        return totalCost;
    }

    template <typename T, typename I>
    void writeToFile(const std::vector<T>& Xaxis, const std::vector<I>& Yaxis, std::filesystem::path filepath)
    {
        std::ofstream file(filepath, std::ios::out);

        if (!file.is_open())
        {
            std::cerr << "Error: Could not open file for writing: " << filepath << std::endl;
            return;
        }

        for (size_t i = 0; i < Xaxis.size(); ++i)
            file << Xaxis[i] << "," << Yaxis[i] << "\n";

        if (file.fail())
            std::cerr << "Error: Failed while writing to file: " << filepath << std::endl;
        else
            std::cout << "File written successfully: " << filepath << std::endl;

        file.close();
    }

    template <typename T, typename I, typename J>
    void writeToFile3(const std::vector<T>& Xaxis,
                      const std::vector<I>& Yaxis,
                      const std::vector<J>& Zaxis,
                      std::filesystem::path filepath)
    {
        std::ofstream file(filepath, std::ios::out);

        if (!file.is_open())
        {
            std::cerr << "Error: Could not open file for writing: " << filepath << std::endl;
            return;
        }

        for (size_t i = 0; i < Xaxis.size(); ++i)
            file << Xaxis[i] << "," << Yaxis[i] << "," << Zaxis[i] << "\n";

        if (file.fail())
            std::cerr << "Error: Failed while writing to file: " << filepath << std::endl;
        else
            std::cout << "File written successfully: " << filepath << std::endl;

        file.close();
    }

    template <typename... Vectors>
    void writeToFileN(const std::filesystem::path& filepath, const Vectors&... vectors)
    {
        std::ofstream file(filepath, std::ios::out);
        if (!file.is_open())
        {
            std::cerr << "Error: Could not open file for writing: " << filepath << std::endl;
            return;
        }

        size_t size = std::min({vectors.size()...}); // take smallest size of vectors

        for (size_t i = 0; i < size; ++i)
        {
            std::ostringstream oss;
            ((oss << vectors[i] << ","), ...);
            std::string line = oss.str();
            file << line.substr(0, line.size() - 1) << "\n";
        }

        if (file.fail())
            std::cerr << "Error: Failed while writing to file: " << filepath << std::endl;
        else
            std::cout << "File written successfully: " << filepath << std::endl;

        file.close();
    }

    std::string fltToStr(float f)
    {
        std::ostringstream oss;
        oss << std::fixed << f;
        std::string str = oss.str();

        // remove zero's
        str.erase(str.find_last_not_of('0') + 1);

        // If it ends with a . remove it too
        if (!str.empty() && str.back() == '.')
        {
            str.pop_back();
        }

        return str;
    }

    void loadPrecomputed(std::vector<TsnePoint2D>& P, std::filesystem::path path)
    {
        std::ifstream in(path, std::ios::binary);
        if (!in)
        {
            throw std::runtime_error("Failed to open file for reading");
        }
        else
        {
            in.read(reinterpret_cast<char*>(P.data()), P.size() * sizeof(TsnePoint2D));
        }
    }

    void storePrecomputed(const std::vector<TsnePoint2D>& P, std::filesystem::path path)
    {
        std::ofstream out(path, std::ios::binary);
        if (!out)
        {
            throw std::runtime_error("Failed to open file for writing");
        }
        else
        {
            out.write(reinterpret_cast<const char*>(P.data()), P.size() * sizeof(TsnePoint2D));
        }
    }
};
