// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "genetic_programming_islands.hpp"

#include <iostream>

#include "initializer/base.hpp"
#include "learning_history.hpp"

#include </usr/lib/clang/20/include/omp.h>
#include <memory>
#include <ostream>

namespace qsr {

GeneticProgrammingIslands::GeneticProgrammingIslands (
    std::shared_ptr<Dataset> dataset, 
    int nislands, 
    int nweights, 
    int npopulation, 
    std::shared_ptr<BaseInitializer> initializer, 
    std::shared_ptr<BaseMutation> mutation, 
    std::shared_ptr<BaseCrossover> crossover, 
    std::shared_ptr<BaseSelection> selection,
    std::shared_ptr<BaseRunnerGenerator> runner_generator) noexcept :
        dataset(dataset),
        initializer(initializer),
        mutation(mutation),
        crossover(crossover),
        selection(selection),
        runner_generator(runner_generator),
        nweights(nweights),
        nislands(nislands),
        npopulation(npopulation)
{
    // Create empty island array
    islands = new GeneticProgramming*[nislands];
}

GeneticProgrammingIslands::~GeneticProgrammingIslands() noexcept 
{
    // Delete island array
    delete[] islands;
}

std::tuple<Expression,std::vector<float>> GeneticProgrammingIslands::fit(int ngenerations, int nsupergenerations, 
                                           int nepochs, float learning_rate, bool verbose) noexcept {
    // Current best solution
    Expression global_best = 0.0;

    // Global learning history
    LearningHistory global_history;

    // Local learning histories
    auto local_histories = new LearningHistory[nislands];

    // Fit
    #pragma omp parallel num_threads(nislands)
    {
        const int threadIdx = omp_get_thread_num();
        const int population_per_island = npopulation / nislands;

        auto runner = runner_generator->generate(dataset, nweights);

        islands[threadIdx] = new GeneticProgramming(
            dataset, 
            nweights, 
            population_per_island, 
            initializer,
            mutation,
            crossover,
            selection,
            runner);
        
        for (int supergeneration = 0; supergeneration < nsupergenerations; ++supergeneration) {
            // Iterate island
            local_histories[threadIdx] = islands[threadIdx]->fit(ngenerations, nepochs, learning_rate);

            // Synchronize all islands
            #pragma omp barrier

            // Migrate solutions
            #pragma omp single
            {
                // Combine learning histories
                LearningHistory history_per_supergeneration;
                for (int i = 0; i < nislands; ++i) {
                    history_per_supergeneration = history_per_supergeneration.combine_with(local_histories[i]);
                }

                // Append to global learning history
                global_history = global_history.concatenate_with(history_per_supergeneration);

                // Find the global best solution
                for (int i = 0; i < nislands; ++i) {
                    const Expression local_best = *islands[i]->get_best_solution();
                    if (local_best.loss < global_best.loss || (supergeneration == 0 && i == 0)) {
                        global_best = local_best;
                    }

                    // Print best result of each island if in verbose mode
                    if (verbose) {
                        std::cout << local_best << " loss=" << local_best.loss << std::endl;
                        if (i == nislands - 1) {
                            std::cout << std::endl;
                        }
                    }
                }

                // Forward best solution of island i to island i+1
                if (nislands > 1) {
                    for (int i = nislands - 1; i >= 0; --i) {
                        const Expression new_best = *islands[i]->get_best_solution();

                        // Replace worst solution of island i+1 with best solution of island i
                        *(islands[(i + 1) % nislands]->get_worst_solution()) = new_best;
                    }
                }
            }

            // Synchronize all islands
            #pragma omp barrier
        }

        delete islands[threadIdx];
    }

    // Get total learning history
    auto hist = global_history.get_learning_history();

    // Return tuple of best solution and learning history
    return std::make_tuple(global_best, hist);
}

}