#include "island.hpp"

#include "../expressions/expression.hpp"
#include "../expressions/binary.hpp"

#include "crossover.hpp"
#include "expression_generator.hpp"
#include "mutation.hpp"

#include <cmath>
#include <iostream>
#include <algorithm>

class ExpressionComparator {
public:
    /* Whether fitness(a) < fitness(b) */
    bool operator() (const Expression& a, const Expression& b) {
        // If a is equal to b, a < b is false.
        if (a == b) {
            return false;
        }

        // If a is NaN, but b is not NaN, treat a as the worst solution.
        if (std::isnan(a.loss) && !std::isnan(b.loss)) {
            return true;
        }

        // If a is not NaN, but b is NaN, treat b as the worst solution
        if (!std::isnan(a.loss) && std::isnan(b.loss)) {
            return false;
        }

        // If both are NaN, NaN < NaN is false
        if (std::isnan(a.loss) && std::isnan(b.loss)) {
            return false;
        }

        // If a is at least 10% worse than b, return true.
        if ((a.loss - b.loss) / b.loss > 0.10) {
            return true;
        }

        // Otherwise, check if a is more complex than b.
        // A simpler expression is preferable to an expression that has just 10% lower loss.
        if (a.num_of_nodes > b.num_of_nodes) {
            return 0;
        }

        // If the expressions have the same complexity (number of nodes), then
        // look at the loss again.
        return a.loss > b.loss;
    }
};

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
        
        std::cout << "Best: " << newbest << " Loss: " << newbest.loss << std::endl;
    }
}