// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "genetic_programming.hpp"

#include "../expressions/expression.hpp"

#include "learning_history.hpp"
#include <cmath>

namespace qsr {

GeneticProgramming::GeneticProgramming(
               std::shared_ptr<const Dataset> dataset, 
               int nweights, 
               int npopulation, 
               int max_depth,
               std::shared_ptr<BaseInitialization> initialization,
               std::shared_ptr<BaseMutation> mutation,
               std::shared_ptr<BaseRecombination> recombination,
               std::shared_ptr<BaseSelection> selection,
               std::shared_ptr<BaseRunner> runner) noexcept : 
               dataset(dataset), 
               nvars(dataset->n), 
               nweights(nweights), 
               npopulation(npopulation % 2 == 0 ? npopulation : npopulation + 1), 
               max_depth(max_depth),
               initialization(initialization),
               mutation(mutation),
               recombination(recombination),
               selection(selection),
               runner(runner)
{
    // Get selector
    selector = selection->get_selector(npopulation);

    // Get mutator
    mutator = mutation->get_mutator(nvars, nweights, max_depth);

    // Get recombiner
    recombiner = recombination->get_recombiner(max_depth);

    // Get initializer
    initializer = initialization->get_initializer(nvars, nweights, npopulation, max_depth);

    // Initialize island with a population of random expressions
    initializer->initialize(population);

    // Fitnesses have never been computed yet
    initialized = false;
}

static auto expression_comparator = [](const Expression& a, const Expression& b) {
    // Return true if a is better (more fit/has less loss) than b

    if (std::isnan(a.loss) && std::isnan(b.loss)) {
        // Return false to be consistent with NaN < NaN, but the result does not actually matter
        return false;
    } 
    else if (std::isnan(a.loss) && !std::isnan(b.loss)) {
        // a is NaN, b is non-NaN; a is worse
        return false;
    }
    else if (!std::isnan(a.loss) && std::isnan(b.loss)) {
        // a is non-NaN, b is NaN; so a is better
        return true;
    } 
    else {
        // Neither is NaN, compare the actual loss values
        return a.loss < b.loss;
    }
};

Expression *GeneticProgramming::get_best_solution() {
    return std::min_element(population.begin(), population.end(), 
    expression_comparator).base();
}

Expression *GeneticProgramming::get_worst_solution() {
    return std::max_element(population.begin(), population.end(), 
    expression_comparator).base();
}

LearningHistory GeneticProgramming::fit(int ngenerations, int nepochs, float learning_rate) noexcept {
    // Create empty learning history
    LearningHistory history;

    // Compute initial fitnesses if not initialized
    if (!initialized) {
        runner->run(population, nepochs, learning_rate);
        initialized = true;
    }

    // Iterate for ngenerations
    for (int generation = 0; generation < ngenerations; ++generation)
    {
        // Get the best solution
        const Expression prev_best = *get_best_solution();

        // Update selection probabilities
        selector->update(&population[0]);

        /* Offspring Generation */
        std::vector<Expression> offspring;
        for (int i = 0; i < npopulation / 2; ++i) {
            const auto &parent1 = selector->select(&population[0]);
            const auto &parent2 = selector->select(&population[0]);
            const auto &children = recombiner->recombine(parent1, parent2);
            const auto &child1 = get<0>(children);
            const auto &child2 = get<1>(children);
            offspring.push_back(mutator->mutate(child1));
            offspring.push_back(mutator->mutate(child2));
        }

        // Replace current population with offspring
        population = offspring;

        // Compute fitnesses
        runner->run(population, nepochs, learning_rate);

        // Find new best
        const Expression new_best = *get_best_solution();

        // Preserve previous best if it is better than the new best
        if (prev_best.loss < new_best.loss) {
            *get_worst_solution() = prev_best;
        }

        // Append to learning history
        history.add_to_history(prev_best.loss < new_best.loss ? prev_best : new_best);
    }

    return history;
}

}