// clang-format off

/*
 *
 * Copyright (c) 2014, Laurens van der Maaten (Delft University of Technology)
 * All rights reserved.
 * (for code that originates from tsne.cpp)
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *    This product includes software developed by the Delft University of Technology.
 * 4. Neither the name of the Delft University of Technology nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY LAURENS VAN DER MAATEN ''AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL LAURENS VAN DER MAATEN BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 *
 */

/*
* Copyright(c)[2019][George Linderman]
* (for code that originates from nbodyfft.cpp)
* 
* MIT License
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

#pragma once

#include <functional>
#include <glm/glm.hpp>
#include <glm/gtx/component_wise.hpp> 
#include <glm/gtx/string_cast.hpp>
#include <utility>
#include <vector>
#include <Fastor/Fastor.h>
#include <boost/sort/sort.hpp>
#include <cstdint>
#include <algorithm>
#include <string>
#include <exception>
#include <thread>
#include <unsupported/Eigen/SparseExtra>
#include <Eigen/Sparse>
#include <iostream>
#include <fstream>
#include <math.h>
#include <float.h>
#include <cstring>
#include <fftw3/fftw3.h>
#include <complex>
#include <chrono>
#include <FMM_tSNE/tsne/solvers/solver.h>
#include <FMM_tSNE/tsne/points/tsne_point_2D.h>
#include <FMM_tSNE/data_loaders/sparse_entry_COO_2D.h>
#include <FMM_tSNE/data_loaders/sparse_entry_CSC_2D.h>

//#include "../../structs/sparseEntry2D.h"
//#include "../../common.h"
//#include "nBodySolver.h"
//#include "../../particles/embeddedPoint.h"
//#include "../../particles/tsnePoint2D.h"
//#include "../../particles/Particle2D.h"
//#include "../../Timer.h"

template <typename T>
class NBodySolverPM : public NBodySolver<T>
{
public:
    unsigned int* C_inp_row_P = nullptr; // row val of sparse matrix
    unsigned int* C_inp_col_P = nullptr; // col val of sparse matrix
    double* C_inp_val_P = nullptr; // scalar val of sparse matrix

    double* C_Y = nullptr; // embedded position array
    int C_N = 0; // number of points
    int C_D = 2; // dimensionality of points

    double* C_dC = nullptr; // derivative array
    int C_n_interpolation_points = 0; // how many interpolation points per box, typically is 4
    //double C_intervals_per_integer = 0.0; // desired side length of the boxes : renamed to cell_size
    int C_min_num_intervals = 0; // minimun amount of boxes per dim

    unsigned int C_nthreads = 1u; // number of threads used

    std::vector<glm::vec2> C_vec_box_lower_bounds;
    std::vector<glm::vec2> C_vec_box_upper_bounds;

    //int iteration_counter = 0;

    NBodySolverPM() :
        C_N(0),
        C_D(2),
        C_nthreads(1u),
        C_n_interpolation_points(0),
        C_min_num_intervals(0),
        C_inp_row_P(nullptr),
        C_inp_col_P(nullptr),
        C_inp_val_P(nullptr),
        C_Y(nullptr),
        C_dC(nullptr) {}

    NBodySolverPM
    (
        Eigen::SparseMatrix<double>& Pmatrix, 
        std::vector<TsnePoint2D>& points, 
        int init_n_interpolation_points, 
        double init_intervals_per_integer, 
        int init_min_num_intervals
    )
    {
        C_nthreads = 1u;

        C_N = points.size();

        C_inp_row_P = new unsigned int[C_N + 1];
        C_inp_col_P = new unsigned int[Pmatrix.nonZeros()];
        C_inp_val_P = new double[Pmatrix.nonZeros()];

        int PmatrixCounter = 0;

        std::vector<SparseEntryCOO2D> sparse_matrix_COO(Pmatrix.nonZeros());
        for (int k = 0; k < Pmatrix.outerSize(); ++k) // https://stackoverflow.com/questions/22421244/eigen-package-iterate-over-row-major-sparse-matrix
        {
            for (Eigen::SparseMatrix<double>::InnerIterator it(Pmatrix, k); it; ++it)
            {
                sparse_matrix_COO[PmatrixCounter] = SparseEntryCOO2D(it.col(), it.row(), it.value());

                PmatrixCounter++;
            }
        }
        std::sort
        (
            sparse_matrix_COO.begin(),
            sparse_matrix_COO.end(),
            [](const SparseEntryCOO2D& a, const SparseEntryCOO2D& b) -> bool
            {
                if (a.row != b.row)
                    return a.row < b.row;
                else
                    return a.col < b.col;
            }
        );
        C_inp_row_P[0] = 0;
        int entryCounter = 0;
        for (int r = 0; r < C_N; r++) // go over each row
        {
            int amount_in_row = 0;

            while (entryCounter < Pmatrix.nonZeros() && sparse_matrix_COO[entryCounter].row == r)
            {
                C_inp_col_P[entryCounter] = sparse_matrix_COO[entryCounter].col;
                C_inp_val_P[entryCounter] = sparse_matrix_COO[entryCounter].val;
                amount_in_row++;
                entryCounter++;
            }

            C_inp_row_P[r + 1] = C_inp_row_P[r] + amount_in_row;
        }

        C_Y = new double[C_N * 2];
        C_dC = new double[C_N * 2];
        for (int i = 0; i < C_N; i++)
        {
            C_Y[i * 2 + 0] = points[i].position.x;
            C_Y[i * 2 + 1] = points[i].position.y;

            C_dC[i * 2 + 0] = 0.0;
            C_dC[i * 2 + 1] = 0.0;
        }

        C_n_interpolation_points = init_n_interpolation_points;
        this->cell_size = init_intervals_per_integer;
        C_min_num_intervals = init_min_num_intervals;
    }

    NBodySolverPM(const NBodySolverPM& other) // copy constructor
    {
        C_N = other.C_N;
        C_D = other.C_D;
        C_nthreads = other.C_nthreads;
        C_n_interpolation_points = other.C_n_interpolation_points;
        this->cell_size = other.cell_size;
        C_min_num_intervals = other.C_min_num_intervals;

        C_inp_row_P = new unsigned int[C_N + 1];
        std::copy(other.C_inp_row_P, other.C_inp_row_P + other.C_N + 1, C_inp_row_P);

        C_inp_col_P = new unsigned int[C_inp_row_P[C_N]];
        std::copy(other.C_inp_col_P, other.C_inp_col_P + other.C_inp_row_P[other.C_N], C_inp_col_P);
        C_inp_val_P = new double[C_inp_row_P[C_N]];
        std::copy(other.C_inp_val_P, other.C_inp_val_P + other.C_inp_row_P[other.C_N], C_inp_val_P);

        C_Y = new double[C_N * 2];
        std::copy(other.C_Y, other.C_Y + 2 * other.C_N, C_Y);
        C_dC = new double[C_N * 2];
        std::copy(other.C_dC, other.C_dC + 2 * other.C_N, C_dC);

        //iteration_counter = other.iteration_counter;
    }

    NBodySolverPM& operator=(const NBodySolverPM& other) // copy assignment operator
    {
        if (this != &other) // self-assignment check
        {
            delete[] C_inp_row_P;
            delete[] C_inp_col_P;
            delete[] C_inp_val_P;

            delete[] C_Y;

            delete[] C_dC;

            C_N = other.C_N;
            C_D = other.C_D;
            C_nthreads = other.C_nthreads;
            C_n_interpolation_points = other.C_n_interpolation_points;
            this->cell_size = other.cell_size;
            C_min_num_intervals = other.C_min_num_intervals;

            C_inp_row_P = new unsigned int[C_N + 1];
            std::copy(other.C_inp_row_P, other.C_inp_row_P + other.C_N + 1, C_inp_row_P);

            C_inp_col_P = new unsigned int[C_inp_row_P[C_N]];
            std::copy(other.C_inp_col_P, other.C_inp_col_P + other.C_inp_row_P[other.C_N], C_inp_col_P);
            C_inp_val_P = new double[C_inp_row_P[C_N]];
            std::copy(other.C_inp_val_P, other.C_inp_val_P + other.C_inp_row_P[other.C_N], C_inp_val_P);

            C_Y = new double[C_N * 2];
            std::copy(other.C_Y, other.C_Y + 2 * other.C_N, C_Y);
            C_dC = new double[C_N * 2];
            std::copy(other.C_dC, other.C_dC + 2 * other.C_N, C_dC);

            //iteration_counter = other.iteration_counter;
        }
        return *this;
    }

    ~NBodySolverPM()
    {
        delete[] C_inp_row_P;
        delete[] C_inp_col_P;
        delete[] C_inp_val_P;

        delete[] C_Y;

        delete[] C_dC;
    }

    void solveNbody(double& total, std::vector<T>& points) override
    {
        for (int i = 0; i < points.size(); i++)
        {
            C_Y[2 * i + 0] = points[i].position.x;
            C_Y[2 * i + 1] = points[i].position.y;
        }

        //std::cout << "running with cell size: " << this->cell_size << "\n";
        computeFftGradient
        (
            nullptr,
            C_inp_row_P,
            C_inp_col_P,
            C_inp_val_P,
            C_Y,
            C_N,
            C_D,
            C_dC,
            C_n_interpolation_points,
            this->cell_size,
            C_min_num_intervals,// make 2
            C_nthreads,
            total
        );


        for (int i = 0; i < C_N; i++)
        {
            points[i].derivative = -glm::dvec2
            (
                C_dC[2 * i + 0],
                C_dC[2 * i + 1]
            );
        }
    }

    void updateTree(std::vector<T>& points, glm::dvec2 minPos, glm::dvec2 maxPos) override
    {

    }

    std::vector<VertexPos2Col3> getNodesBufferData(int nodeLevelToShow) override
    {
        std::vector<VertexPos2Col3> result;

        if (nodeLevelToShow != 0)
        {
            for (int i = 0; i < C_vec_box_lower_bounds.size(); i++)
            {
                glm::vec2 lower = C_vec_box_lower_bounds[i];
                glm::vec2 upper = C_vec_box_upper_bounds[i];
                glm::vec2 xlyu = glm::vec2(lower.x, upper.y);
                glm::vec2 xuyl = glm::vec2(upper.x, lower.y);

                glm::vec3 color(1.0f);
            
                result.push_back(VertexPos2Col3(lower, color));
                result.push_back(VertexPos2Col3(xlyu, color));

                result.push_back(VertexPos2Col3(lower, color));
                result.push_back(VertexPos2Col3(xuyl, color));

                result.push_back(VertexPos2Col3(xlyu, color));
                result.push_back(VertexPos2Col3(upper, color));

                result.push_back(VertexPos2Col3(xuyl, color));
                result.push_back(VertexPos2Col3(upper, color));
            }
        }

        return result;
    }

private:
    // ----------------------- small stuff -----------------------------------------------------

    double custom_HUGE_ENUF = 1e300;
    float custom_INFINITY = ((float)(custom_HUGE_ENUF * custom_HUGE_ENUF));

    static double squared_cauchy_2d(double x1, double x2, double y1, double y2, double df)
    {
        return pow(1.0 + pow(x1 - y1, 2) + pow(x2 - y2, 2), -2);
    }

    typedef double (*kernel_type)(double, double, double);

    typedef double (*kernel_type_2d)(double, double, double, double, double);








    // ----------------------- helper helper function - interpolate -----------------------------------------------------


    void interpolate(int n_interpolation_points, int N, const double* y_in_box, const double* y_tilde_spacings, double* interpolated_values)
    {
        // The denominators are the same across the interpolants, so we only need to compute them once
        auto* denominator = new double[n_interpolation_points];
        for (int i = 0; i < n_interpolation_points; i++)
        {
            denominator[i] = 1;
            for (int j = 0; j < n_interpolation_points; j++)
            {
                if (i != j) {
                    denominator[i] *= y_tilde_spacings[i] - y_tilde_spacings[j];
                }
            }
        }
        // Compute the numerators and the interpolant value
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < n_interpolation_points; j++)
            {
                interpolated_values[j * N + i] = 1;
                for (int k = 0; k < n_interpolation_points; k++)
                {
                    if (j != k) {
                        interpolated_values[j * N + i] *= y_in_box[i] - y_tilde_spacings[k];
                    }
                }
                interpolated_values[j * N + i] /= denominator[j];
            }
        }

        delete[] denominator;
    }




    // ----------------------- helper function - n_body_fft_2d -----------------------------------------------------

    void n_body_fft_2d(int N, int n_terms, double* xs, double* ys, double* chargesQij, int n_boxes,
        int n_interpolation_points, double* box_lower_bounds, double* box_upper_bounds,
        double* y_tilde_spacings, std::complex<double>* fft_kernel_tilde, double* potentialQij, unsigned int nthreads)
    {
        int n_total_boxes = n_boxes * n_boxes;
        int total_interpolation_points = n_total_boxes * n_interpolation_points * n_interpolation_points;

        double coord_min = box_lower_bounds[0];
        double box_width = box_upper_bounds[0] - box_lower_bounds[0];

        auto* point_box_idx = new int[N];

        // Determine which box each point belongs to
        for (int i = 0; i < N; i++)
        {
            auto x_idx = static_cast<int>((xs[i] - coord_min) / box_width);
            auto y_idx = static_cast<int>((ys[i] - coord_min) / box_width);
            // TODO: Figure out how on earth x_idx can be less than zero...
            // It's probably something to do with the fact that we use the single lowest coord for both dims? Probably not
            // this, more likely negative 0 if rounding errors
            if (x_idx >= n_boxes)
            {
                x_idx = n_boxes - 1;
            }
            else if (x_idx < 0)
            {
                x_idx = 0;
            }

            if (y_idx >= n_boxes)
            {
                y_idx = n_boxes - 1;
            }
            else if (y_idx < 0)
            {
                y_idx = 0;
            }
            point_box_idx[i] = y_idx * n_boxes + x_idx;
        }

        // Compute the relative position of each point in its box in the interval [0, 1]
        auto* x_in_box = new double[N];
        auto* y_in_box = new double[N];
        for (int i = 0; i < N; i++)
        {
            int box_idx = point_box_idx[i];
            double x_min = box_lower_bounds[box_idx];
            double y_min = box_lower_bounds[n_total_boxes + box_idx];
            x_in_box[i] = (xs[i] - x_min) / box_width;
            y_in_box[i] = (ys[i] - y_min) / box_width;
        }

        //INITIALIZE_TIME
        //START_TIME

        /*
         * Step 1: Interpolate kernel using Lagrange polynomials and compute the w coefficients
         */
         // Compute the interpolated values at each real point with each Lagrange polynomial in the `x` direction
        auto* x_interpolated_values = new double[N * n_interpolation_points];
        interpolate(n_interpolation_points, N, x_in_box, y_tilde_spacings, x_interpolated_values);
        // Compute the interpolated values at each real point with each Lagrange polynomial in the `y` direction
        auto* y_interpolated_values = new double[N * n_interpolation_points];
        interpolate(n_interpolation_points, N, y_in_box, y_tilde_spacings, y_interpolated_values);

        auto* w_coefficients = new double[total_interpolation_points * n_terms]();
        for (int i = 0; i < N; i++)
        {
            int box_idx = point_box_idx[i];
            int box_j = box_idx / n_boxes;
            int box_i = box_idx % n_boxes;
            for (int interp_i = 0; interp_i < n_interpolation_points; interp_i++)
            {
                for (int interp_j = 0; interp_j < n_interpolation_points; interp_j++)
                {
                    // Compute the index of the point in the interpolation grid of points
                    int idx = (box_i * n_interpolation_points + interp_i) * (n_boxes * n_interpolation_points) + (box_j * n_interpolation_points) + interp_j;
                    for (int d = 0; d < n_terms; d++)
                    {
                        w_coefficients[idx * n_terms + d] +=
                            y_interpolated_values[interp_j * N + i] *
                            x_interpolated_values[interp_i * N + i] *
                            chargesQij[i * n_terms + d];
                    }
                }
            }
        }

        //END_TIME("Step 1");
        //START_TIME;
    /*
     * Step 2: Compute the values v_{m, n} at the equispaced nodes, multiply the kernel matrix with the coefficients w
     */
        auto* y_tilde_values = new double[total_interpolation_points * n_terms]();
        int n_fft_coeffs_half = n_interpolation_points * n_boxes;
        int n_fft_coeffs = 2 * n_interpolation_points * n_boxes;
        auto* mpol_sort = new double[total_interpolation_points];

        // FFT of fft_input
        auto* fft_input = new double[n_fft_coeffs * n_fft_coeffs]();
        auto* fft_w_coefficients = new std::complex<double>[n_fft_coeffs * (n_fft_coeffs / 2 + 1)];
        auto* fft_output = new double[n_fft_coeffs * n_fft_coeffs]();

        fftw_plan plan_dft, plan_idft;
        plan_dft = fftw_plan_dft_r2c_2d(n_fft_coeffs, n_fft_coeffs, fft_input, reinterpret_cast<fftw_complex*>(fft_w_coefficients), FFTW_ESTIMATE);
        plan_idft = fftw_plan_dft_c2r_2d(n_fft_coeffs, n_fft_coeffs, reinterpret_cast<fftw_complex*>(fft_w_coefficients), fft_output, FFTW_ESTIMATE);

        for (int d = 0; d < n_terms; d++)
        {
            for (int i = 0; i < total_interpolation_points; i++)
            {
                mpol_sort[i] = w_coefficients[i * n_terms + d];
            }

            for (int i = 0; i < n_fft_coeffs_half; i++)
            {
                for (int j = 0; j < n_fft_coeffs_half; j++)
                {
                    fft_input[i * n_fft_coeffs + j] = mpol_sort[i * n_fft_coeffs_half + j];
                }
            }

            fftw_execute(plan_dft);

            // Take the Hadamard product of two complex vectors
            for (int i = 0; i < n_fft_coeffs * (n_fft_coeffs / 2 + 1); i++)
            {
                double x_ = fft_w_coefficients[i].real();
                double y_ = fft_w_coefficients[i].imag();
                double u_ = fft_kernel_tilde[i].real();
                double v_ = fft_kernel_tilde[i].imag();
                fft_w_coefficients[i].real(x_ * u_ - y_ * v_);
                fft_w_coefficients[i].imag(x_ * v_ + y_ * u_);
            }

            // Invert the computed values at the interpolated nodes
            fftw_execute(plan_idft);
            for (int i = 0; i < n_fft_coeffs_half; i++)
            {
                for (int j = 0; j < n_fft_coeffs_half; j++)
                {
                    int row = n_fft_coeffs_half + i;
                    int col = n_fft_coeffs_half + j;

                    // FFTW doesn't perform IDFT normalization, so we have to do it ourselves. This is done by dividing
                    // the result with the number of points in the input
                    mpol_sort[i * n_fft_coeffs_half + j] = fft_output[row * n_fft_coeffs + col] / (double)(n_fft_coeffs * n_fft_coeffs);
                }
            }
            for (int i = 0; i < n_fft_coeffs_half * n_fft_coeffs_half; i++)
            {
                y_tilde_values[i * n_terms + d] = mpol_sort[i];
            }
        }

        fftw_destroy_plan(plan_dft);
        fftw_destroy_plan(plan_idft);
        delete[] fft_w_coefficients;
        delete[] fft_input;
        delete[] fft_output;
        delete[] mpol_sort;
        //END_TIME("FFT");
        //START_TIME
        /*
         * Step 3: Compute the potentials \tilde{\phi}
         */
         //PARALLEL_FOR(nthreads, N,
         //    {
         //        int box_idx = point_box_idx[loop_i];
         //        int box_i = box_idx % n_boxes;
         //        int box_j = box_idx / n_boxes;
         //        for (int interp_i = 0; interp_i < n_interpolation_points; interp_i++)
         //        {
         //            for (int interp_j = 0; interp_j < n_interpolation_points; interp_j++)
         //            {
         //                for (int d = 0; d < n_terms; d++)
         //                {
         //                    // Compute the index of the point in the interpolation grid of points
         //                    int idx = (box_i * n_interpolation_points + interp_i) * (n_boxes * n_interpolation_points) + (box_j * n_interpolation_points) + interp_j;
         //                    potentialQij[loop_i * n_terms + d] +=
         //                            x_interpolated_values[interp_i * N + loop_i] *
         //                            y_interpolated_values[interp_j * N + loop_i] *
         //                            y_tilde_values[idx * n_terms + d];
         //                }
         //            }
         //        }
         //    });
        {
            //if (nthreads > 1)
            //{
            //    std::vector<std::thread> threads(nthreads);
            //    for (int t = 0; t < nthreads; t++)
            //    {
            //        threads[t] = std::thread
            //        (
            //            std::bind
            //            (
            //                [&](const int bi, const int ei, const int t)
            //                {
            //                    for (int loop_i = bi;loop_i < ei;loop_i++)
            //                    {
            //                        {
            //                            int box_idx = point_box_idx[loop_i];
            //                            int box_i = box_idx % n_boxes;
            //                            int box_j = box_idx / n_boxes;
            //                            for (int interp_i = 0; interp_i < n_interpolation_points; interp_i++)
            //                            {
            //                                for (int interp_j = 0; interp_j < n_interpolation_points; interp_j++)
            //                                {
            //                                    for (int d = 0; d < n_terms; d++)
            //                                    {
            //                                        // Compute the index of the point in the interpolation grid of points
            //                                        int idx = (box_i * n_interpolation_points + interp_i) * (n_boxes * n_interpolation_points) + (box_j * n_interpolation_points) + interp_j;
            //                                        potentialQij[loop_i * n_terms + d] +=
            //                                            x_interpolated_values[interp_i * N + loop_i] *
            //                                            y_interpolated_values[interp_j * N + loop_i] *
            //                                            y_tilde_values[idx * n_terms + d];
            //                                    }
            //                                }
            //                            }
            //                        };
            //                    }
            //                },
            //                t * N / nthreads,
            //                    (t + 1) == nthreads ? N : (t + 1) * N / nthreads,
            //                    t
            //                    )
            //        );
            //    }
            //    std::for_each(threads.begin(), threads.end(), [](std::thread& x) {x.join();});
            //}
            //else
            {
                for (int loop_i = 0; loop_i < N; loop_i++)
                {
                    {
                        int box_idx = point_box_idx[loop_i];
                        int box_i = box_idx % n_boxes;
                        int box_j = box_idx / n_boxes;
                        for (int interp_i = 0; interp_i < n_interpolation_points; interp_i++)
                        {
                            for (int interp_j = 0; interp_j < n_interpolation_points; interp_j++)
                            {
                                for (int d = 0; d < n_terms; d++)
                                {
                                    // Compute the index of the point in the interpolation grid of points
                                    int idx = (box_i * n_interpolation_points + interp_i) * (n_boxes * n_interpolation_points) + (box_j * n_interpolation_points) + interp_j;
                                    potentialQij[loop_i * n_terms + d] +=
                                        x_interpolated_values[interp_i * N + loop_i] *
                                        y_interpolated_values[interp_j * N + loop_i] *
                                        y_tilde_values[idx * n_terms + d];
                                }
                            }
                        }
                    };
                }
            }
        }



        //END_TIME("Step 3");
        delete[] point_box_idx;
        delete[] x_interpolated_values;
        delete[] y_interpolated_values;
        delete[] w_coefficients;
        delete[] y_tilde_values;
        delete[] x_in_box;
        delete[] y_in_box;
    }

    // ----------------------- helper function - precompute_2d -----------------------------------------------------

    void precompute_2d(double x_max, double x_min, double y_max, double y_min, int n_boxes, int n_interpolation_points,
        kernel_type_2d kernel, double* box_lower_bounds, double* box_upper_bounds, double* y_tilde_spacings,
        double* y_tilde, double* x_tilde, std::complex<double>* fft_kernel_tilde, double df)
    {
        /*
         * Set up the boxes
         */
        int n_total_boxes = n_boxes * n_boxes;
        double box_width = (x_max - x_min) / (double)n_boxes;

        // Left and right bounds of each box, first the lower bounds in the x direction, then in the y direction
        for (int i = 0; i < n_boxes; i++)
        {
            for (int j = 0; j < n_boxes; j++)
            {
                box_lower_bounds[i * n_boxes + j] = j * box_width + x_min;
                box_upper_bounds[i * n_boxes + j] = (j + 1) * box_width + x_min;

                box_lower_bounds[n_total_boxes + i * n_boxes + j] = i * box_width + y_min;
                box_upper_bounds[n_total_boxes + i * n_boxes + j] = (i + 1) * box_width + y_min;
            }
        }

        // Coordinates of each (equispaced) interpolation node for a single box
        double h = 1 / (double)n_interpolation_points;
        y_tilde_spacings[0] = h / 2;
        for (int i = 1; i < n_interpolation_points; i++)
        {
            y_tilde_spacings[i] = y_tilde_spacings[i - 1] + h;
        }

        // Coordinates of all the equispaced interpolation points
        int n_interpolation_points_1d = n_interpolation_points * n_boxes;
        int n_fft_coeffs = 2 * n_interpolation_points_1d;

        h = h * box_width;
        x_tilde[0] = x_min + h / 2;
        y_tilde[0] = y_min + h / 2;
        for (int i = 1; i < n_interpolation_points_1d; i++)
        {
            x_tilde[i] = x_tilde[i - 1] + h;
            y_tilde[i] = y_tilde[i - 1] + h;
        }

        /*
         * Evaluate the kernel at the interpolation nodes and form the embedded generating kernel vector for a circulant
         * matrix
         */
        auto* kernel_tilde = new double[n_fft_coeffs * n_fft_coeffs]();
        for (int i = 0; i < n_interpolation_points_1d; i++)
        {
            for (int j = 0; j < n_interpolation_points_1d; j++)
            {
                double tmp = kernel(y_tilde[0], x_tilde[0], y_tilde[i], x_tilde[j], df);
                kernel_tilde[(n_interpolation_points_1d + i) * n_fft_coeffs + (n_interpolation_points_1d + j)] = tmp;
                kernel_tilde[(n_interpolation_points_1d - i) * n_fft_coeffs + (n_interpolation_points_1d + j)] = tmp;
                kernel_tilde[(n_interpolation_points_1d + i) * n_fft_coeffs + (n_interpolation_points_1d - j)] = tmp;
                kernel_tilde[(n_interpolation_points_1d - i) * n_fft_coeffs + (n_interpolation_points_1d - j)] = tmp;
            }
        }

        // Precompute the FFT of the kernel generating matrix
        fftw_plan p = fftw_plan_dft_r2c_2d(n_fft_coeffs, n_fft_coeffs, kernel_tilde, reinterpret_cast<fftw_complex*>(fft_kernel_tilde), FFTW_ESTIMATE);\
        fftw_execute(p);



        fftw_destroy_plan(p);
        delete[] kernel_tilde;
    }




    // ----------------------- main function - computeFftGradient -----------------------------------------------------

    // Compute the gradient of the t-SNE cost function using the FFT interpolation based approximation
    void computeFftGradient(double* P, unsigned int* inp_row_P, unsigned int* inp_col_P, double* inp_val_P, double* Y, int N, int D, double* dC,
        int n_interpolation_points,
        double intervals_per_integer,
        int min_num_intervals,
        unsigned int nthreads,
        double& total)
    {
        //fftw_init_threads();
        //fftw_plan_with_nthreads(1);
        // std::cout << "fftw_planner_nthreads(void): " << fftw_planner_nthreads() << std::endl;
        // P
        // inp_row_P
        // inp_col_P
        // inp_val_P

        // Y array of size N * D with positions of the embedded points
        // N number of points
        // D dimension amount of embedded space
        // dc array size N * D with the gradient


        // n_interpolation_points how many interpolation points per box, typically is 4
        // intervals_per_integer desired side length of the boxes
        // min_num_intervals minimun amount of boxes per dim

        // nthreads how many threads to use

        // Zero out the gradient
        for (unsigned long i = 0; i < N * D; i++) dC[i] = 0.0;

        // For convenience, split the x and y coordinate values
        auto* xs = new double[N];
        auto* ys = new double[N];

        double min_coord = custom_INFINITY;
        double max_coord = -custom_INFINITY;
        // Find the min/max values of the x and y coordinates
        for (unsigned long i = 0; i < N; i++)
        {
            xs[i] = Y[i * 2 + 0];
            ys[i] = Y[i * 2 + 1];
            if (xs[i] > max_coord) max_coord = xs[i];
            else if (xs[i] < min_coord) min_coord = xs[i];
            if (ys[i] > max_coord) max_coord = ys[i];
            else if (ys[i] < min_coord) min_coord = ys[i];
        }

        // The number of "charges" or s+2 sums i.e. number of kernel sums
        int n_terms = 4;
        auto* chargesQij = new double[N * n_terms];
        auto* potentialsQij = new double[N * n_terms]();

        // Prepare the terms that we'll use to compute the sum i.e. the repulsive forces
        for (unsigned long j = 0; j < N; j++)
        {
            chargesQij[j * n_terms + 0] = 1;
            chargesQij[j * n_terms + 1] = xs[j];
            chargesQij[j * n_terms + 2] = ys[j];
            chargesQij[j * n_terms + 3] = xs[j] * xs[j] + ys[j] * ys[j];
        }

        // Compute the number of boxes in a single dimension and the total number of boxes in 2d
        //auto n_boxes_per_dim = static_cast<int>(fmax(min_num_intervals, (max_coord - min_coord) / intervals_per_integer));
        //auto n_boxes_per_dim = static_cast<int>(min_num_intervals);
        auto n_boxes_per_dim = static_cast<int>((max_coord - min_coord) / intervals_per_integer);

        // FFTW works faster on numbers that can be written as  2^a 3^b 5^c 7^d
        // 11^e 13^f, where e+f is either 0 or 1, and the other exponents are
        // arbitrary
        //int allowed_n_boxes_per_dim[20] = { 25,36, 50, 55, 60, 65, 70, 75, 80, 85, 90, 96, 100, 110, 120, 130, 140,150, 175, 200 };
        //if (n_boxes_per_dim < allowed_n_boxes_per_dim[19])
        //{
        //    //Round up to nearest grid point
        //    int chosen_i;
        //    for (chosen_i = 0; allowed_n_boxes_per_dim[chosen_i] < n_boxes_per_dim; chosen_i++);
        //    n_boxes_per_dim = allowed_n_boxes_per_dim[chosen_i];
        //}

        //n_boxes_per_dim = min_num_intervals; // delete this for extra performance!!!!!!!!!!!!!!!!!!!!!!!!
        n_boxes_per_dim = std::max(4, n_boxes_per_dim);
        //std::cout << "n_boxes_per_dim: " << n_boxes_per_dim << std::endl;

        int n_boxes = n_boxes_per_dim * n_boxes_per_dim;

        auto* box_lower_bounds = new double[2 * n_boxes];
        auto* box_upper_bounds = new double[2 * n_boxes];
        auto* y_tilde_spacings = new double[n_interpolation_points];
        int n_interpolation_points_1d = n_interpolation_points * n_boxes_per_dim;
        auto* x_tilde = new double[n_interpolation_points_1d]();
        auto* y_tilde = new double[n_interpolation_points_1d]();
        auto* fft_kernel_tilde = new std::complex<double>[2 * n_interpolation_points_1d * 2 * n_interpolation_points_1d];


        //INITIALIZE_TIME;
        //START_TIME;
        precompute_2d(max_coord, min_coord, max_coord, min_coord, n_boxes_per_dim, n_interpolation_points,
            &squared_cauchy_2d,
            box_lower_bounds, box_upper_bounds, y_tilde_spacings, x_tilde, y_tilde, fft_kernel_tilde, 1.0);
        n_body_fft_2d(N, n_terms, xs, ys, chargesQij, n_boxes_per_dim, n_interpolation_points, box_lower_bounds,
            box_upper_bounds, y_tilde_spacings, fft_kernel_tilde, potentialsQij, nthreads);

        C_vec_box_lower_bounds.clear();
        C_vec_box_lower_bounds.resize(n_boxes);
        C_vec_box_upper_bounds.clear();
        C_vec_box_upper_bounds.resize(n_boxes);
        for (int i = 0; i < n_boxes; i++)
        {
            C_vec_box_lower_bounds[i] = glm::vec2
            (
                box_lower_bounds[i],
                box_lower_bounds[n_boxes + i]
            );
            C_vec_box_upper_bounds[i] = glm::vec2
            (
                box_upper_bounds[i],
                box_upper_bounds[n_boxes + i]
            );
        }

        // Compute the normalization constant Z or sum of q_{ij}. This expression is different from the one in the original
        // paper, but equivalent. This is done so we need only use a single kernel (K_2 in the paper) instead of two
        // different ones. We subtract N at the end because the following sums over all i, j, whereas Z contains i \neq j
        double sum_Q = 0;
        for (unsigned long i = 0; i < N; i++)
        {
            double phi1 = potentialsQij[i * n_terms + 0];
            double phi2 = potentialsQij[i * n_terms + 1];
            double phi3 = potentialsQij[i * n_terms + 2];
            double phi4 = potentialsQij[i * n_terms + 3];

            sum_Q += (1 + xs[i] * xs[i] + ys[i] * ys[i]) * phi1 - 2 * (xs[i] * phi2 + ys[i] * phi3) + phi4;
        }
        sum_Q -= N;
        total = sum_Q;
        //this->current_sum_Q = sum_Q;

        //double* pos_f = new double[N * 2];

        //END_TIME("Total Interpolation");
            //START_TIME;
        // Now, figure out the Gaussian component of the gradient. This corresponds to the "attraction" term of the
        // gradient. It was calculated using a fast KNN approach, so here we just use the results that were passed to this
        // function
        //PARALLEL_FOR
        //(
        //    nthreads,
        //    N,
        //    {
        //        double dim1 = 0;
        //        double dim2 = 0;

        //        for (unsigned int i = inp_row_P[loop_i]; i < inp_row_P[loop_i + 1]; i++)
        //        {
        //            // Compute pairwise distance and Q-value
        //            unsigned int ind3 = inp_col_P[i];
        //            double d_ij = (xs[loop_i] - xs[ind3]) * (xs[loop_i] - xs[ind3]) + (ys[loop_i] - ys[ind3]) * (ys[loop_i] - ys[ind3]);
        //            double q_ij = 1 / (1 + d_ij);

        //            dim1 += inp_val_P[i] * q_ij * (xs[loop_i] - xs[ind3]);
        //            dim2 += inp_val_P[i] * q_ij * (ys[loop_i] - ys[ind3]);
        //        }
        //        pos_f[loop_i * 2 + 0] = dim1;
        //        pos_f[loop_i * 2 + 1] = dim2;
        //    }
        //);
        
        //{
        //    if (nthreads > 1)
        //    {
        //        std::vector<std::thread> threads(nthreads);
        //        for (int t = 0; t < nthreads; t++)
        //        {
        //            threads[t] = std::thread
        //            (
        //                std::bind
        //                (
        //                    [&](const int bi, const int ei, const int t)
        //                    {
        //                        for (int loop_i = bi;loop_i < ei;loop_i++)
        //                        {
        //                            {
        //                                double dim1 = 0;
        //                                double dim2 = 0;

        //                                for (unsigned int i = inp_row_P[loop_i]; i < inp_row_P[loop_i + 1]; i++)
        //                                {
        //                                    // Compute pairwise distance and Q-value
        //                                    unsigned int ind3 = inp_col_P[i];
        //                                    double d_ij = (xs[loop_i] - xs[ind3]) * (xs[loop_i] - xs[ind3]) + (ys[loop_i] - ys[ind3]) * (ys[loop_i] - ys[ind3]);
        //                                    double q_ij = 1 / (1 + d_ij);

        //                                    dim1 += inp_val_P[i] * q_ij * (xs[loop_i] - xs[ind3]);
        //                                    dim2 += inp_val_P[i] * q_ij * (ys[loop_i] - ys[ind3]);
        //                                }
        //                                pos_f[loop_i * 2 + 0] = dim1;
        //                                pos_f[loop_i * 2 + 1] = dim2;
        //                            };
        //                        }
        //                    },
        //                    t * N / nthreads,
        //                        (t + 1) == nthreads ? N : (t + 1) * N / nthreads,
        //                        t
        //                        )
        //            );
        //        }
        //        std::for_each(threads.begin(), threads.end(), [](std::thread& x) {x.join();});
        //    }
        //    else
        //    {
        //        Timer timer;

        //        for (int loop_i = 0; loop_i < N; loop_i++)
        //        {
        //            {
        //                double dim1 = 0;
        //                double dim2 = 0;

        //                for (unsigned int i = inp_row_P[loop_i]; i < inp_row_P[loop_i + 1]; i++)
        //                {
        //                    // Compute pairwise distance and Q-value
        //                    unsigned int ind3 = inp_col_P[i];
        //                    double d_ij = (xs[loop_i] - xs[ind3]) * (xs[loop_i] - xs[ind3]) + (ys[loop_i] - ys[ind3]) * (ys[loop_i] - ys[ind3]);
        //                    double q_ij = 1 / (1 + d_ij);

        //                    dim1 += inp_val_P[i] * q_ij * (xs[loop_i] - xs[ind3]);
        //                    dim2 += inp_val_P[i] * q_ij * (ys[loop_i] - ys[ind3]);
        //                }
        //                pos_f[loop_i * 2 + 0] = dim1;
        //                pos_f[loop_i * 2 + 1] = dim2;
        //            };
        //        }

        //        timer.endTimer("PM attractive");
        //    }
        //}





        //END_TIME("Attractive Forces");
        //printf("Attractive forces took %lf\n", (diff(start20,end20))/(double)1E6);







        // Make the negative term, or F_rep in the equation 3 of the paper
        //double* neg_f = new double[N * 2];
        for (unsigned int i = 0; i < N; i++)
        {
            //neg_f[i * 2 + 0] = (xs[i] * potentialsQij[i * n_terms] - potentialsQij[i * n_terms + 1]) / sum_Q;
            //neg_f[i * 2 + 1] = (ys[i] * potentialsQij[i * n_terms] - potentialsQij[i * n_terms + 2]) / sum_Q;

            //float exageration = iteration_counter < 250 ? 4.0f : 1.0f;
            //dC[i * 2 + 0] = 4.0f * (exageration * pos_f[i * 2] - neg_f[i * 2]);
            //dC[i * 2 + 1] = 4.0f * (exageration * pos_f[i * 2 + 1] - neg_f[i * 2 + 1]);

            dC[i * 2 + 0] = -(xs[i] * potentialsQij[i * n_terms] - potentialsQij[i * n_terms + 1]);
            dC[i * 2 + 1] = -(ys[i] * potentialsQij[i * n_terms] - potentialsQij[i * n_terms + 2]);
        }

        //delete[] pos_f;
        //delete[] neg_f;
        delete[] potentialsQij;
        delete[] chargesQij;
        delete[] xs;
        delete[] ys;
        delete[] box_lower_bounds;
        delete[] box_upper_bounds;
        delete[] y_tilde_spacings;
        delete[] y_tilde;
        delete[] x_tilde;
        delete[] fft_kernel_tilde;
    }
};
