// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef GENETIC_PROGRAMMING_ISLANDS_HPP
#define GENETIC_PROGRAMMING_ISLANDS_HPP

#include "../dataset/dataset.hpp"

#include "common/toolbox.hpp"
#include "genetic_programming.hpp"

#include "../runners/runner_generator_base.hpp"

#include <memory>

namespace qsr {

class GeneticProgrammingIslands {
public:
    GeneticProgrammingIslands(int nislands, const Config &config, const Toolbox &toolbox,
                              std::shared_ptr<BaseRunnerGenerator> runner_generator) noexcept;

    std::tuple<Expression,std::vector<float>> fit(std::shared_ptr<Dataset> dataset, 
                                                  int ngenerations, int nsupergenerations, 
                                                  int nepochs, float learning_rate, 
                                                  bool verbose = false) noexcept;

    ~GeneticProgrammingIslands() noexcept;

private:
    /// Genetic operators shared by all islands
    const Toolbox toolbox;

    /// Runner generator shared by all islands
    const std::shared_ptr<BaseRunnerGenerator> runner_generator;

    /// Number of islands
    const int nislands;

    /// Array of islands
    GeneticProgramming **islands;

    /// Global configuration (contains total population size and total offspring size)
    Config global_config;

    /// Local configuration (contains per island population size and per island offspring size)
    Config local_config;
};

}

#endif