// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include </usr/lib/clang/20/include/omp.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

#include "../genetic/genetic_programming_islands.hpp"

#include "../genetic/mutation/default.hpp"
#include "../genetic/crossover/default.hpp"
#include "../genetic/selection/fitness_proportional_selection.hpp"

namespace py = pybind11;

void fit(py::array_t<float> numpy_X, py::array_t<float> numpy_y, 
        int nweights,
        int npopulation, 
        int ngenerations, int nsupergenerations,
        int max_initial_depth,
        int nthreads,
        BaseMutation *mutation,
        BaseCrossover *crossover,
        BaseSelection *selection) 
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

    // Run genetic programming
    GeneticProgrammingIslands gp(
        dataset, nweights, npopulation, 
        max_initial_depth, nthreads, 
        ngenerations, nsupergenerations, 
        *mutation, *crossover, *selection);

    gp.iterate();

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

    m.def("fit", &fit, "Fits a symbolic expression to given feature matrix X and target vector y",
        py::arg("X"),                               // Required argument
        py::arg("y"),                               // Required argument
        py::arg("nweights") = 2,                    // Optional argument with default value
        py::arg("npopulation") = 1000,              // Optional argument with default value
        py::arg("ngenerations") = 10,               // Optional argument with default value
        py::arg("nsupergenerations") = 2,           // Optional argument with default value
        py::arg("max_initial_depth") = 3,           // Optional argument with default value
        py::arg("nthreads") = 1,                    // Optional argument with default value
        py::arg("mutation"),                        // Required argument
        py::arg("crossover"),                       // Required argument
        py::arg("selection")                        // Required argument
    );

    py::class_<BaseMutation>(m, "BaseMutation")
        .def(py::init<>());

    py::class_<DefaultMutation, BaseMutation>(m, "DefaultMutation")
        .def(py::init<int, int, int, float>(),
            py::arg("nvars"), 
            py::arg("nweights"), 
            py::arg("max_depth") = 3, 
            py::arg("mutation_probability") = 0.7);

    py::class_<BaseCrossover>(m, "BaseCrossover")
        .def(py::init<>());

    py::class_<DefaultCrossover, BaseCrossover>(m, "DefaultCrossover")
        .def(py::init<float>(), 
             py::arg("crossover_probability") = 0.7);

    py::class_<BaseSelection>(m, "BaseSelection")
        .def(py::init<>());

    py::class_<FitnessProportionalSelection, BaseSelection>(m, "FitnessProportionalSelection")
        .def(py::init<>());
}
