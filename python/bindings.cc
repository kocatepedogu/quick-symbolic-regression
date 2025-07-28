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

void fit(Dataset *dataset, 
        int nweights,
        int npopulation, 
        int ngenerations, int nsupergenerations,
        int max_initial_depth,
        int nthreads,
        BaseMutation *mutation,
        BaseCrossover *crossover,
        BaseSelection *selection) 
{
    // Run genetic programming
    GeneticProgrammingIslands gp(
        *dataset, nweights, npopulation, 
        max_initial_depth, nthreads, 
        ngenerations, nsupergenerations, 
        *mutation, *crossover, *selection);

    gp.iterate();
}

PYBIND11_MODULE(libquicksr, m) {
    m.doc() = "pybind11 libquicksr plugin";

    m.def("fit", &fit, "Fits a symbolic expression to given feature matrix X and target vector y",
        py::arg("dataset"),                         // Required argument
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

    py::class_<Dataset>(m, "Dataset")
        .def(py::init<py::array_t<float>, py::array_t<float>>(),
            py::arg("X"),
            py::arg("y"));
}
