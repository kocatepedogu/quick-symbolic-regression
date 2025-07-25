#include "island.hpp"

#include "../expressions/expression.hpp"

#include "crossover.hpp"
#include "expression_comparator.hpp"
#include "expression_generator.hpp"
#include "mutation.hpp"

#include <algorithm>

Island::Island(const Dataset& dataset, 
               int nweights, 
               int npopulation, 
               int max_initial_depth, 
               int max_mutation_depth, 
               float mutation_probability,
               float crossover_probability) noexcept : 
               dataset(dataset), 
               nvars(dataset.n), 
               nweights(nweights), 
               npopulation(npopulation % 2 == 0 ? npopulation : npopulation + 1), 
               runner(dataset, nweights), 
               mutator(dataset.n, nweights, max_mutation_depth, mutation_probability),
               crossover(crossover_probability)
{
    // Initialize island with a population of random expressions
    ExpressionGenerator initial_expression_generator(nvars, nweights, max_initial_depth);
    for (int i = 0; i < npopulation; ++i) {
        population.push_back(initial_expression_generator.generate());
    }

    // Compute initial fitnesses
    runner.run(population, 10);

    // Sort population with respect to fitness
    std::sort(population.begin(), population.end(), ExpressionComparator());
}

Expression Island::get_best_solution() {
    return population[npopulation - 1];
}

void Island::insert_solution(Expression e) {
    population[npopulation - 2] = e;
}

void Island::iterate(int niters) noexcept {
    for (int iter = 0; iter < niters; ++iter)
    {
        /* Parent Selection */
        const auto &best = population[npopulation - 1];
        const auto &nextbest = population[npopulation - 2];

        /* Offspring Generation */
        std::vector<Expression> offspring;
        for (int i = 0; i < npopulation / 2; ++i) {
            const auto &children = crossover.crossover(best, nextbest);
            const auto &child1 = get<0>(children);
            const auto &child2 = get<1>(children);
            offspring.push_back(mutator.mutate(child1));
            offspring.push_back(mutator.mutate(child2));
        }

        // Replace current population with offspring
        population = offspring;

        // Compute fitnesses
        runner.run(population, 1);

        // Sort population
        std::sort(population.begin(), population.end(), ExpressionComparator());

        /* Survival Selection */
        Expression &newbest = population[npopulation - 1];
        if (!ExpressionComparator()(best, newbest)) {
            newbest = best;
        }
    }
}