// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include </usr/lib/clang/20/include/omp.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

#include "../genetic/genetic_programming.hpp"
#include "../genetic/expression_comparator.hpp"

#include "../genetic/mutation/default.hpp"
#include "../genetic/crossover/default.hpp"
#include "../genetic/selection/fitness_proportional_selection.hpp"

namespace py = pybind11;

void fit(py::array_t<float> numpy_X, py::array_t<float> numpy_y, 
        int nweights,
        int npopulation, 
        int ngenerations, int nsupergenerations,
        int max_initial_depth, int max_mutation_depth,
        float mutation_probability,
        int nthreads) 
{
    auto numpy_X_buffer_info = numpy_X.request();
    auto numpy_y_buffer_info = numpy_y.request();

    if (numpy_X_buffer_info.ndim != 1 && numpy_X_buffer_info.ndim != 2) {
        throw py::value_error("Argument X (feature matrix) must be an either one- or two-dimensional array.");
    }

    if (numpy_y_buffer_info.ndim != 1) {
        throw py::value_error("Argument y (target vector) must be a one-dimensional array.");
    }

    if (numpy_X_buffer_info.shape[0] != numpy_y_buffer_info.shape[0]) {
        throw py::value_error("First dimensions of X and y must match (the number of data points)");
    }

    // Number of rows / data points
    ssize_t num_data_points = numpy_X_buffer_info.shape[0];

    // Number of columns / features
    ssize_t num_features = numpy_X_buffer_info.ndim == 1 ? 1 : numpy_X_buffer_info.shape[1];

    // Allocate and fill X
    float **X = (float **)malloc(num_data_points * sizeof *X);
    for (int i = 0; i < num_data_points; ++i) {
        X[i] = (float *)malloc(num_features * sizeof **X);
        for (int j = 0; j < num_features; ++j) {
            X[i][j] = (static_cast<float*>(numpy_X_buffer_info.ptr))[i * num_features + j];
        }
    }

    // Allocate and fill y
    float *y = (float *)malloc(num_data_points * sizeof *y);
    for (int i = 0; i < num_data_points; ++i) {
        y[i] = (static_cast<float*>(numpy_y_buffer_info.ptr))[i];
    }

    // Create dataset
    Dataset dataset(X, y, num_data_points, num_features);

    // Create empty island array
    GeneticProgramming **islands = new GeneticProgramming*[nthreads];

    // Current best solution
    Expression best = 0.0;

    // Fit
    #pragma omp parallel num_threads(nthreads)
    {
        const int threadIdx = omp_get_thread_num();
        const int population_per_thread = npopulation / nthreads;

        DefaultMutation mutation(num_features, nweights, max_mutation_depth, mutation_probability);
        DefaultCrossover crossover(1.0);
        FitnessProportionalSelection selection(population_per_thread);

        islands[threadIdx] = new GeneticProgramming(
            dataset, 
            nweights, 
            population_per_thread, 
            max_initial_depth,
            mutation,
            crossover,
            selection);

        for (int supergeneration = 0; supergeneration < nsupergenerations; ++supergeneration) {
            // Iterate island
            islands[threadIdx]->iterate(ngenerations);

            // Synchronize all islands
            #pragma omp barrier

            // Migrate solutions
            #pragma omp single
            {
                // Find the best solution
                for (int i = 0; i < nthreads; ++i) {
                    Expression new_best = islands[i]->get_best_solution();
                    if (ExpressionComparator()(best, new_best) || (supergeneration == 0 && i == 0)) {
                        best = new_best;
                    }
                }

                // Forward best solution of island i to island i+1
                std::cerr << std::endl;
                for (int i = nthreads - 1; i >= 0; --i) {
                    Expression new_best = islands[i]->get_best_solution();

                    int next = (i + 1) % nthreads;
                    islands[next]->insert_solution(new_best);

                    // Print best result of each island
                    std::cerr << new_best << std::endl;
                }
            }

            // Synchronize all islands
            #pragma omp barrier
        }

        delete islands[threadIdx];
    }

    delete[] islands;

    // Free y
    free(y);

    // Free X
    for (int i = 0; i < num_data_points; ++i) {
        free(X[i]);
    }
    free(X);
}

PYBIND11_MODULE(libquicksr, m) {
    m.doc() = "pybind11 libquicksr plugin";
    m.def("fit", &fit, "Fits a symbollic expression to given feature matrix X and target vector y",
        py::arg("X"),                               // Required argument
        py::arg("y"),                               // Required argument
        py::arg("nweights") = 2,                    // Optional argument with default value
        py::arg("npopulation") = 1000,              // Optional argument with default value
        py::arg("ngenerations") = 10,               // Optional argument with default value
        py::arg("nsupergenerations") = 2,           // Optional argument with default value
        py::arg("max_initial_depth") = 3,           // Optional argument with default value
        py::arg("max_mutation_depth") = 3,          // Optional argument with default value
        py::arg("max_mutation_probability") = 0.7,  // Optional argument with default value
        py::arg("nthreads") = 1                     // Optional argument with default value
    );
}
