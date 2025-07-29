// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include </usr/lib/clang/20/include/omp.h>

#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "../genetic/genetic_programming_islands.hpp"

#include "../genetic/initializer/default.hpp"
#include "../genetic/mutation/default.hpp"
#include "../genetic/crossover/default.hpp"
#include "../genetic/selection/fitness_proportional_selection.hpp"
#include "../runners/inter-individual/runner_generator.hpp"
#include "../runners/intra-individual/runner_generator.hpp"

namespace py = pybind11;

PYBIND11_MODULE(libquicksr, m) {
    m.doc() = "pybind11 libquicksr plugin";

    /* Mutation Classes */

    py::class_<BaseMutation, std::shared_ptr<BaseMutation>>(m, "BaseMutation")
        .def(py::init<>());

    py::class_<DefaultMutation, BaseMutation, std::shared_ptr<DefaultMutation>>(m, "DefaultMutation")
        .def(py::init<int, int, int, float>(),
            py::arg("nvars"), 
            py::arg("nweights"), 
            py::arg("max_depth") = 3, 
            py::arg("mutation_probability") = 0.7);

    /* Crossover Classes */

    py::class_<BaseCrossover, std::shared_ptr<BaseCrossover>>(m, "BaseCrossover")
        .def(py::init<>());

    py::class_<DefaultCrossover, BaseCrossover, std::shared_ptr<DefaultCrossover>>(m, "DefaultCrossover")
        .def(py::init<float>(), 
             py::arg("crossover_probability") = 0.8);

    /* Selection Classes */

    py::class_<BaseSelection, std::shared_ptr<BaseSelection>>(m, "BaseSelection")
        .def(py::init<>());

    py::class_<FitnessProportionalSelection, BaseSelection, std::shared_ptr<FitnessProportionalSelection>>(m, "FitnessProportionalSelection")
        .def(py::init<>());

    /* Initialization Classes */

    py::class_<BaseInitializer, std::shared_ptr<BaseInitializer>>(m, "BaseInitializer")
        .def(py::init<>());

    py::class_<DefaultInitializer, BaseInitializer, std::shared_ptr<DefaultInitializer>>(m, "DefaultInitializer")
        .def(py::init<int, int, int, int>(),
             py::arg("nvars"),
             py::arg("nweights"),
             py::arg("max_depth") = 1,
             py::arg("npopulation"));

    /* Dataset Class */

    py::class_<Dataset, std::shared_ptr<Dataset>>(m, "Dataset")
        .def(py::init<py::array_t<float>, py::array_t<float>>(),
            py::arg("X"),
            py::arg("y"));

    /* Runner Generator Classes */

    py::class_<BaseRunnerGenerator, std::shared_ptr<BaseRunnerGenerator>>(m, "BaseRunnerGenerator")
        .def(py::init<>());

    py::class_<inter_individual::RunnerGenerator, BaseRunnerGenerator, std::shared_ptr<inter_individual::RunnerGenerator>>(m, "InterIndividualRunnerGenerator")
        .def(py::init<>());
        
    py::class_<intra_individual::RunnerGenerator, BaseRunnerGenerator, std::shared_ptr<intra_individual::RunnerGenerator>>(m, "IntraIndividualRunnerGenerator")
        .def(py::init<>());

    /* Genetic Programming Algorithm Classes */

    py::class_<GeneticProgrammingIslands>(m, "GeneticProgrammingIslands")
        // Constructor
        .def(py::init<
            std::shared_ptr<Dataset>, 
            int, int, int,
            std::shared_ptr<BaseInitializer>,
            std::shared_ptr<BaseMutation>,
            std::shared_ptr<BaseCrossover>,
            std::shared_ptr<BaseSelection>,
            std::shared_ptr<BaseRunnerGenerator>>(),
            py::arg("dataset"),
            py::arg("nislands"),
            py::arg("nweights"),
            py::arg("npopulation"),
            py::arg("initializer"),
            py::arg("mutation"),
            py::arg("crossover"),
            py::arg("selection"),
            py::arg("runner_generator"))

        // Fit method
        .def("fit", &GeneticProgrammingIslands::fit,
            py::arg("ngenerations"),
            py::arg("nsupergenerations"),
            py::arg("nepochs") = 1,
            py::arg("learning_rate") = 1e-3,
            py::arg("verbose") = false);
}
