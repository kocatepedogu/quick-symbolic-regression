// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef GENETIC_PROGRAMMING_ISLANDS_HPP
#define GENETIC_PROGRAMMING_ISLANDS_HPP

#include "../dataset/dataset.hpp"

#include "recombination/base.hpp"
#include "genetic_programming.hpp"
#include "initialization/base.hpp"
#include "mutation/base.hpp"
#include "selection/base.hpp"
#include "../runners/runner_generator_base.hpp"

namespace qsr {

class GeneticProgrammingIslands {
public:
    GeneticProgrammingIslands(std::shared_ptr<Dataset> dataset, 
                              int nislands, 
                              int nweights, 
                              int npopulation, 
                              std::shared_ptr<BaseInitialization> initialization,
                              std::shared_ptr<BaseMutation> mutation,
                              std::shared_ptr<BaseRecombination> recombination,
                              std::shared_ptr<BaseSelection> selection,
                              std::shared_ptr<BaseRunnerGenerator> runner_generator) noexcept;

    ~GeneticProgrammingIslands() noexcept;

    std::tuple<Expression,std::vector<float>> fit(int ngenerations, int nsupergenerations, int nepochs, float learning_rate, bool verbose = false) noexcept;

private:
    /// Dataset shared by all islands
    std::shared_ptr<Dataset> dataset;

    /// Population initializer shared by all islands
    std::shared_ptr<BaseInitialization> initialization;

    /// Mutation operator shared by all islands
    std::shared_ptr<BaseMutation> mutation;

    /// Crossover operator shared by all islands
    std::shared_ptr<BaseRecombination> recombination;

    /// Selection operator shared by all islands
    std::shared_ptr<BaseSelection> selection;

    /// Runner generator shared by all islands
    std::shared_ptr<BaseRunnerGenerator> runner_generator;

    /// Properties of solutions
    const int nweights;

    /// Total population size
    const int npopulation;

    /// Array of islands
    GeneticProgramming **islands;

    /// Number of islands
    const int nislands;
};

}

#endif