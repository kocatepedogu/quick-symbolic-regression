// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "genetic/genetic_programming_islands.hpp"
#include "genetic/learning_history.hpp"

#include </usr/lib/clang/20/include/omp.h>
#include <memory>
#include <ostream>
#include <iostream>

namespace qsr {

GeneticProgrammingIslands::GeneticProgrammingIslands (
    int nislands, const Config &config, const Toolbox &toolbox,
    std::shared_ptr<BaseRunnerGenerator> runner_generator) noexcept :
        toolbox(toolbox), 
        runner_generator(runner_generator), 
        nislands(nislands) {

    // Create local configuration from global configuration
    // Let population size and offspring size be the size per island instead of the total size
    const Config local_config = Config(
        config.nvars,
        config.nweights, 
        config.max_depth, 
        config.npopulation / nislands, 
        config.noffspring / nislands, 
        config.function_set);

    // Initialize islands
    islands = new GeneticProgramming*[nislands];
    for (int i = 0; i < nislands; ++i) {
        islands[i] = new GeneticProgramming(local_config, toolbox, 
            runner_generator->generate(local_config.nweights));
    }

    // Initialize local learning histories
    local_learning_history = new LearningHistory[nislands];

    // Let the initial best solution be null
    global_best = nullptr;
}

GeneticProgrammingIslands::~GeneticProgrammingIslands() noexcept {
    // Delete islands
    for (int i = 0; i < nislands; ++i) {
        delete islands[i];
    }
    delete[] islands;

    // Delete local learning histories
    delete[] local_learning_history;
}

std::tuple<Expression,std::vector<float>> GeneticProgrammingIslands::fit(
    std::shared_ptr<Dataset> dataset, 
    int ngenerations, int nsupergenerations, 
    int nepochs, float learning_rate, bool verbose) noexcept 
{
    for (int supergeneration = 0; supergeneration < nsupergenerations; ++supergeneration) {
        // Evolve each island separately on a different CPU core
        #pragma omp parallel for 
        for (int island_idx = 0; island_idx < nislands; ++island_idx) {
            local_learning_history[island_idx] = islands[island_idx]->fit(dataset, ngenerations, nepochs, learning_rate);

            // Print the status of islands that are completed
            if (verbose){ display_local_status (island_idx); }
        }

        // Combine local learning histories and concatenate with the global learning history
        update_learning_history();

        // Update global best solution
        update_global_best();

        // Display evolution state
        if (verbose) { display_global_status(); }

        // Migrate solutions between islands
        migrate_solutions();
    }

    // Return tuple of best solution and learning history
    return std::make_tuple(*global_best, global_learning_history.get_learning_history());
}

void GeneticProgrammingIslands::migrate_solutions() noexcept {
    // Forward best solution of island i to island i+1
    if (nislands > 1) {
        for (int i = nislands - 1; i >= 0; --i) {
            const Expression new_best = *islands[i]->get_best_solution();

            // Replace worst solution of island i+1 with best solution of island i
            *(islands[(i + 1) % nislands]->get_worst_solution()) = new_best;
        }
    }
}

void GeneticProgrammingIslands::update_global_best() noexcept {
    // Iterate over all islands
    for (int i = 0; i < nislands; ++i) {
        const Expression local_best = *islands[i]->get_best_solution();

        // Update the global best solution if the local best solution is better
        if (global_best == nullptr || local_best.loss < global_best->loss) {
            global_best = std::make_shared<Expression>(local_best);
        }
    }
}

void GeneticProgrammingIslands::update_learning_history() noexcept {
    // Combine learning histories
    LearningHistory history_per_supergeneration;
    for (int i = 0; i < nislands; ++i) {
        history_per_supergeneration = history_per_supergeneration.combine_with(local_learning_history[i]);
    }

    // Append to global learning history
    global_learning_history = global_learning_history.concatenate_with(history_per_supergeneration);
}

void GeneticProgrammingIslands::display_local_status(int island_idx) noexcept {
    const Expression *best_solution = islands[island_idx]->get_best_solution();

    std::cout << "Island " << island_idx;
    std::cout << " ";

    std::cout << "Best solution: " << *best_solution;
    std::cout << " ";

    std::cout << "Loss: " << best_solution->loss;
    std::cout << std::endl;
}

void GeneticProgrammingIslands::display_global_status() noexcept {
    std::cout << "Global best solution: " << *global_best;
    std::cout << " ";

    std::cout << "Loss: " << global_best->loss;
    std::cout << std::endl;
    std::cout << std::endl;
}

}