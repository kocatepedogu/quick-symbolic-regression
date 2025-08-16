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
               int noffspring,
               int max_depth,
               std::shared_ptr<BaseInitialization> initialization,
               std::shared_ptr<BaseMutation> mutation,
               std::shared_ptr<BaseRecombination> recombination,
               std::shared_ptr<BaseSelection> selection,
               std::shared_ptr<BaseRunner> runner,
               std::shared_ptr<FunctionSet> function_set) noexcept : 
               dataset(dataset), 
               nvars(dataset->n), 
               nweights(nweights), 
               npopulation(npopulation % 2 == 0 ? npopulation : npopulation + 1), 
               noffspring(noffspring % 2 == 0 ? noffspring : noffspring + 1),
               max_depth(max_depth),
               initialization(initialization),
               mutation(mutation),
               recombination(recombination),
               selection(selection),
               runner(runner),
               function_set(function_set)
{
    // Create config
    Config config(nvars, nweights, max_depth, npopulation, function_set);

    // Get selector
    selector = selection->get_selector(npopulation);

    // Get mutator
    mutator = mutation->get_mutator(config);

    // Get recombiner
    recombiner = recombination->get_recombiner(max_depth);

    // Get initializer
    initializer = initialization->get_initializer(config);

    // Initialize island with a population of random expressions
    initializer->initialize(population);

    // Fitnesses have never been computed yet
    initialized = false;
}

Expression *GeneticProgramming::get_best_solution() {
    return std::min_element(population.begin(), population.end(), std::greater<Expression>()).base();
}

Expression *GeneticProgramming::get_worst_solution() {
    return std::max_element(population.begin(), population.end(), std::greater<Expression>()).base();
}

static void calculate_fitnesses(std::vector<Expression> &pop) {
    // Compute fitnesses from losses
    for (int i = 0; i < pop.size(); ++i) {
        const float loss = pop[i].loss;
        
        if (std::isnan(loss) && !std::isfinite(loss)) {
            pop[i].fitness = 0.0f;
        }
        else if (loss == 0.0f) {
            pop[i].fitness = std::numeric_limits<float>::max();
        }
        else {
            pop[i].fitness = 1 / loss;
        }
    }

    // Find the mean fitness and the mean number of nodes
    float mean_fitness = 0.0;
    float mean_length = 0.0;
    for (const auto &expr : pop) {
        mean_fitness += expr.fitness / (float)pop.size();
        mean_length += expr.num_of_nodes / (float)pop.size();
    }

    // Find the covariance of fitness and number of nodes
    float covariance = 0.0;
    for (const auto &expr : pop) {
        covariance += (expr.fitness - mean_fitness) * (expr.num_of_nodes - mean_length) / (float)pop.size();
    }

    // Find the variance of number of nodes
    double variance_length = 0.0;
    for (const auto &expr : pop) {
        variance_length += (expr.num_of_nodes - mean_length) * (expr.num_of_nodes - mean_length) / (float)pop.size();
    }

    // Find the parsimony pressure coefficient
    double ct = covariance / variance_length;

    // Subtract from the fitness of each function the parsimony pressure term
    for (auto &expr : pop) {
        expr.fitness -= ct * expr.num_of_nodes;
    }
}

LearningHistory GeneticProgramming::fit(int ngenerations, int nepochs, float learning_rate) noexcept {
    // Create empty learning history
    LearningHistory history;

    // Compute initial fitnesses if not initialized
    if (!initialized) {
        runner->run(population, nepochs, learning_rate);
        calculate_fitnesses(population);
        initialized = true;
    }

    // Iterate for ngenerations
    for (int generation = 0; generation < ngenerations; ++generation)
    {
        // Update selection probabilities
        selector->update(&population[0]);

        /* Offspring Generation */
        std::vector<Expression> offspring;
        for (int i = 0; i < noffspring / 2; ++i) {
            const auto &parent1 = selector->select(&population[0]);
            const auto &parent2 = selector->select(&population[0]);
            const auto &children = recombiner->recombine(parent1, parent2);
            const auto &child1 = get<0>(children);
            const auto &child2 = get<1>(children);
            offspring.push_back(mutator->mutate(child1));
            offspring.push_back(mutator->mutate(child2));
        }

        // Compute losses
        runner->run(offspring, nepochs, learning_rate);

        // Insert offspring into population
        population.insert(population.end(), offspring.begin(), offspring.end());

        // Calculate fitnesses
        calculate_fitnesses(population);

        // Sort population by fitness in descending order
        std::sort(population.begin(), population.end(), std::greater<Expression>());

        // Remove the worst half of the population
        population.resize(npopulation);

        // Find new best
        const Expression best = *get_best_solution();

        // Append to learning history
        history.add_to_history(best);
    }

    return history;
}

}