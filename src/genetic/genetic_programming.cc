// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "genetic/genetic_programming.hpp"

#include "expressions/expression.hpp"
#include "genetic/common/toolbox.hpp"
#include "genetic/learning_history.hpp"

#include <cmath>

namespace qsr {

GeneticProgramming::GeneticProgramming(const Config &config, const Toolbox &toolbox, std::shared_ptr<BaseRunner> runner) 
    noexcept : config(config), runner(runner)
{
    // Ensure population size and offspring size are even
    if (config.npopulation % 2 != 0) {
        this->config.npopulation++;
    }

    // Get selector
    selector = toolbox.selection->get_selector(ceil(this->config.npopulation * config.survival_rate));

    // Get mutator
    mutator = toolbox.mutation->get_mutator(this->config);

    // Get recombiner
    recombiner = toolbox.recombination->get_recombiner(this->config.max_depth);

    // Get initializer
    auto initializer = toolbox.initialization->get_initializer(this->config);

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

static void calculate_fitnesses(std::vector<Expression> &pop, bool enable_parsimony_pressure) {
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

    if (enable_parsimony_pressure) {
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
}

LearningHistory GeneticProgramming::fit(std::shared_ptr<const Dataset> dataset, int ngenerations, int nepochs, float learning_rate) noexcept {
    // Create empty learning history
    LearningHistory history;

    // Compute initial fitnesses if not initialized
    if (!initialized) {
        // Compute losses
        runner->run(population, dataset, nepochs, learning_rate);

        // Calculate fitnesses
        calculate_fitnesses(population, config.enable_parsimony_pressure);

        // Save initial best loss to learning history
        history.add_to_history(*get_best_solution());

        // Set initialization flag to true
        initialized = true;
    }

    int elitism_count = ceil(config.elite_rate * (float)config.npopulation);

    // Iterate for ngenerations
    for (int generation = 0; generation < ngenerations; ++generation)
    {
        // Sort population by fitness in descending order
        std::sort(population.begin(), population.end(), std::greater<Expression>());

        // Update selection probabilities
        selector->update(&population[0]);

        /* Offspring Generation */
        std::vector<Expression> offspring;
        offspring.reserve(config.npopulation);

        /* Copy the elite directly to offspring */
        for (int i = 0; i < elitism_count; ++i) {
            offspring.push_back(population[i]);
        }

        /* Generate the rest from parents */
        while (offspring.size() < config.npopulation) {
            const auto &parent1 = selector->select(&population[0]);
            const auto &parent2 = selector->select(&population[0]);

            const auto &children = recombiner->recombine(parent1, parent2);
            const auto &child1 = get<0>(children);
            const auto &child2 = get<1>(children);

            offspring.push_back(mutator->mutate(child1));
            if (offspring.size() < config.npopulation) {
                offspring.push_back(mutator->mutate(child2));
            }
        }

        // Compute losses
        runner->run(offspring, dataset, nepochs, learning_rate);

        // Calculate fitnesses
        calculate_fitnesses(offspring, config.enable_parsimony_pressure);

        // Let new population be the offspring
        population = offspring;

        // Find new best
        const Expression best = *get_best_solution();

        // Append to learning history
        history.add_to_history(best);
    }

    return history;
}

}