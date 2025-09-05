// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "genetic/common/function_set.hpp"
#include "genetic/genetic_programming_islands.hpp"

#include "genetic/initialization/grow_initialization.hpp"
#include "genetic/initialization/full_initialization.hpp"
#include "genetic/initialization/ramped_half_and_half_initialization.hpp"

#include "genetic/selection/fitness_proportional_selection.hpp"
#include "genetic/selection/rank_selection.hpp"

#include "genetic/mutation/subtree_mutation.hpp"
#include "genetic/mutation/hoist_mutation.hpp"
#include "genetic/mutation/point_mutation.hpp"
#include "genetic/mutation/distribution_mutation.hpp"

#include "genetic/recombination/default.hpp"

#include "runners/inter-individual/runner_generator.hpp"
#include "runners/intra-individual/runner_generator.hpp"
#include "runners/cpu/runner_generator.hpp"
#include "runners/hybrid/runner_generator.hpp"

namespace py = pybind11;

namespace qsr {

PYBIND11_MODULE(libquicksr, m) {
    m.doc() = "pybind11 libquicksr plugin";

    /* Mutation Classes */

    py::class_<BaseMutation, std::shared_ptr<BaseMutation>>(m, "BaseMutation")
        .def(py::init<>());

    py::class_<SubtreeMutation, BaseMutation, std::shared_ptr<SubtreeMutation>>(m, "SubtreeMutation")
        .def(py::init<int, float>(),
            py::arg("max_depth_increment") = 3, 
            py::arg("mutation_probability") = 0.7);

    py::class_<HoistMutation, BaseMutation, std::shared_ptr<HoistMutation>>(m, "HoistMutation")
        .def(py::init<float>(),
            py::arg("mutation_probability") = 0.7);

    py::class_<PointMutation, BaseMutation, std::shared_ptr<PointMutation>>(m, "PointMutation")
        .def(py::init<float>(),
            py::arg("mutation_probability") = 0.7);

    py::class_<DistributionMutation, BaseMutation, std::shared_ptr<DistributionMutation>>(m, "DistributionMutation")
        .def(py::init<std::vector<std::shared_ptr<BaseMutation>>, std::vector<float>>(),
            py::arg("mutations"),
            py::arg("probabilities"));

    /* Recombination Classes */

    py::class_<BaseRecombination, std::shared_ptr<BaseRecombination>>(m, "BaseRecombination")
        .def(py::init<>());

    py::class_<DefaultRecombination, BaseRecombination, std::shared_ptr<DefaultRecombination>>(m, "DefaultRecombination")
        .def(py::init<float>(), 
             py::arg("crossover_probability") = 0.8);

    /* Selection Classes */

    py::class_<BaseSelection, std::shared_ptr<BaseSelection>>(m, "BaseSelection")
        .def(py::init<>());

    py::class_<FitnessProportionalSelection, BaseSelection, std::shared_ptr<FitnessProportionalSelection>>(m, "FitnessProportionalSelection")
        .def(py::init<>());

    py::class_<RankSelection, BaseSelection, std::shared_ptr<RankSelection>>(m, "RankSelection")
        .def(py::init<float>(), 
             py::arg("sp") = 1.0f);

    /* Initialization Classes */

    py::class_<BaseInitialization, std::shared_ptr<BaseInitialization>>(m, "BaseInitialization")
        .def(py::init<>());

    py::class_<GrowInitialization, BaseInitialization, std::shared_ptr<GrowInitialization>>(m, "GrowInitialization")
        .def(py::init<std::optional<int>>(),
             py::arg("init_depth") = std::nullopt);

    py::class_<FullInitialization, BaseInitialization, std::shared_ptr<FullInitialization>>(m, "FullInitialization")
        .def(py::init<std::optional<int>>(),
             py::arg("init_depth") = std::nullopt);

    py::class_<RampedHalfAndHalfInitialization, BaseInitialization, std::shared_ptr<RampedHalfAndHalfInitialization>>(m, "RampedHalfAndHalfInitialization")
        .def(py::init<std::optional<int>>(),
             py::arg("init_depth") = std::nullopt);

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

    py::class_<cpu::RunnerGenerator, BaseRunnerGenerator, std::shared_ptr<cpu::RunnerGenerator>>(m, "CPURunnerGenerator")
        .def(py::init<>());

    py::class_<hybrid::RunnerGenerator, BaseRunnerGenerator, std::shared_ptr<hybrid::RunnerGenerator>>(m, "HybridRunnerGenerator")
        .def(py::init<>());

    /* Function Set Class */

    py::class_<FunctionSet, std::shared_ptr<FunctionSet>>(m, "FunctionSet")
        .def(py::init<std::vector<std::string>>(),
            py::arg("functions"));

    /* Config Class */

    py::class_<Config>(m, "Config")
        .def(py::init<int, int, int, int, int, std::shared_ptr<FunctionSet>>(),
            py::arg("nvars"),
            py::arg("nweights"),
            py::arg("max_depth"),
            py::arg("npopulation"),
            py::arg("noffspring"),
            py::arg("function_set"));

    /* Toolbox Class */

    py::class_<Toolbox>(m, "Toolbox")
        .def(py::init<
            std::shared_ptr<BaseInitialization>,
            std::shared_ptr<BaseMutation>,
            std::shared_ptr<BaseRecombination>,
            std::shared_ptr<BaseSelection>>(),
            py::arg("initialization"),
            py::arg("mutation"),
            py::arg("recombination"),
            py::arg("selection"));

    /* Genetic Programming Algorithm Classes */

    py::class_<GeneticProgrammingIslands>(m, "GeneticProgrammingIslands")
        // Constructor
        .def(py::init<
            int, const Config &, const Toolbox &,
            std::shared_ptr<BaseRunnerGenerator>>(),
            py::arg("nislands"),
            py::arg("config"),
            py::arg("toolbox"),
            py::arg("runner_generator"))

        // Fit method
        .def("fit", &GeneticProgrammingIslands::fit,
            py::arg("dataset"),
            py::arg("ngenerations"),
            py::arg("nsupergenerations"),
            py::arg("nepochs") = 1,
            py::arg("learning_rate") = 1e-3,
            py::arg("verbose") = false);

    /* Expression Class */

    py::enum_<operation_t>(m, "Operation")
        .value("CONSTANT", operation_t::CONSTANT)
        .value("PARAMETER", operation_t::PARAMETER)
        .value("IDENTITY", operation_t::IDENTITY)
        .value("ADDITION", operation_t::ADDITION)
        .value("SUBTRACTION", operation_t::SUBTRACTION)
        .value("MULTIPLICATION", operation_t::MULTIPLICATION)
        .value("DIVISION", operation_t::DIVISION)
        .value("SINE", operation_t::SINE)
        .value("COSINE", operation_t::COSINE)
        .value("EXPONENTIAL", operation_t::EXPONENTIAL)
        .value("RECTIFIED_LINEAR_UNIT", operation_t::RECTIFIED_LINEAR_UNIT);

    py::class_<Expression>(m, "Expression")
        .def_readwrite("operation", &Expression::operation)
        .def_readwrite("value", &Expression::value)
        .def_readwrite("argindex", &Expression::argindex)
        .def_readwrite("operands", &Expression::operands)
        .def_readwrite("num_of_nodes", &Expression::num_of_nodes)
        .def_readwrite("loss", &Expression::loss)
        .def_readwrite("weights", &Expression::weights)
        
        .def("__repr__", [](const Expression& expr) {
            std::ostringstream oss; oss << expr;
            return oss.str();
        });
}

}
