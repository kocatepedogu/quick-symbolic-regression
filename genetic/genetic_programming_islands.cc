#include "genetic_programming_islands.hpp"

#include <iostream>

#include "expression_comparator.hpp"
#include "initializer/base.hpp"

GeneticProgrammingIslands::GeneticProgrammingIslands (
    const Dataset& dataset, 
    int nweights, 
    int npopulation,
    int nislands, 
    int ngenerations, 
    int nsupergenerations,
    BaseInitializer& initializer, 
    BaseMutation& mutation, 
    BaseCrossover& crossover, 
    BaseSelection& selection) noexcept :
        dataset(dataset),
        initializer(initializer),
        mutation(mutation),
        crossover(crossover),
        selection(selection),
        nislands(nislands),
        nweights(nweights),
        npopulation(npopulation),
        ngenerations(ngenerations),
        nsupergenerations(nsupergenerations)
{
    // Create empty island array
    islands = new GeneticProgramming*[nislands];
}

GeneticProgrammingIslands::~GeneticProgrammingIslands() noexcept 
{
    // Delete island array
    delete[] islands;
}

Expression GeneticProgrammingIslands::iterate() noexcept {
    // Current best solution
    Expression best = 0.0;

    // Fit
    #pragma omp parallel num_threads(nislands)
    {
        const int threadIdx = omp_get_thread_num();
        const int population_per_island = npopulation / nislands;

        islands[threadIdx] = new GeneticProgramming(
            dataset, 
            nweights, 
            population_per_island, 
            initializer,
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
                for (int i = 0; i < nislands; ++i) {
                    Expression new_best = islands[i]->get_best_solution();
                    if (ExpressionComparator()(best, new_best) || (supergeneration == 0 && i == 0)) {
                        best = new_best;
                    }
                }

                // Forward best solution of island i to island i+1
                std::cerr << std::endl;
                for (int i = nislands - 1; i >= 0; --i) {
                    Expression new_best = islands[i]->get_best_solution();

                    int next = (i + 1) % nislands;
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

    return best;
}