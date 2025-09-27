// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef GENETIC_PROGRAMMING_ISLANDS_HPP
#define GENETIC_PROGRAMMING_ISLANDS_HPP

#include "genetic/common/toolbox.hpp"
#include "genetic/genetic_programming.hpp"
#include "runners/runner_generator_base.hpp"
#include "dataset/dataset.hpp"

#include <memory>

namespace qsr {

class GeneticProgrammingIslands {
public:
    GeneticProgrammingIslands(
        int nislands, const Config &config, const Toolbox &toolbox,
        std::shared_ptr<BaseRunnerGenerator> runner_generator) noexcept;

    std::tuple<Expression,std::vector<double>,std::vector<double>,std::vector<long>> fit(
        std::shared_ptr<Dataset> dataset, 
        int ngenerations, int nsupergenerations, 
        int nepochs, double learning_rate,
        bool verbose = false) noexcept;

    ~GeneticProgrammingIslands() noexcept;

private:
    /**
     * @brief Runner generator shared by all islands
     */
    const std::shared_ptr<BaseRunnerGenerator> runner_generator;

    /** 
      * @brief Genetic operators shared by all islands
      */
    const Toolbox toolbox;

    /**
     * @brief Global configuration
     */
    Config config;

    /** 
      * @brief Number of islands
      */
    const int nislands;

    /**
     * @brief Number of HIP streams/states
     */
    const int nstreams;

    /**
     *  @brief HIP streams for islands
     *  Number of hip streams can be fewer than number of islands
     */
    HIPState **hipStreams;

    /** 
      * @brief Array of islands
      */
    GeneticProgramming **islands;

    /**
     * @brief Best solution out of all islands at any particular generation
     */ 
    LearningHistory global_learning_history;

    /**
     * @brief Best solution of each island at any particular generation
     */
    LearningHistory *local_learning_history;

    /**
     * @brief Best solution ever found
     */
    std::shared_ptr<Expression> global_best;

    /**
     * @brief Updates the global and local learning histories
     */
    void update_learning_history() noexcept;

    /**
     * @brief Updates the global best solution by comparing the best solutions of all islands
     */
    void update_global_best() noexcept;

    /**
     * @brief Migrates solutions between islands
     */
    void migrate_solutions() noexcept;

    /**
     * @brief Displays the best solution the given island
     */
    void display_local_status(int island_idx) noexcept;

    /**
     * @brief Displays the global best solution
     */
    void display_global_status() noexcept;
};

}

#endif