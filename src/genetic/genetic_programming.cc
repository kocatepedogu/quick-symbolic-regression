// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "genetic/genetic_programming.hpp"

#include "expressions/expression.hpp"
#include "genetic/common/toolbox.hpp"
#include "genetic/learning_history.hpp"
#include "util/precision.hpp"

#include <cmath>

namespace qsr {
extern int nislands;

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
        const double loss = pop[i].loss;
        
        if (std::isnan(loss) && std::isnan(loss)) {
            pop[i].fitness = 0.0;
        }
        else if (loss == 0.0) {
            pop[i].fitness = std::numeric_limits<double>::max();
        }
        else {
            pop[i].fitness = 1 / loss;
        }
    }

    if (enable_parsimony_pressure) {
        // Find the mean fitness and the mean number of nodes
        double mean_fitness = 0.0;
        double mean_length = 0.0;
        for (const auto &expr : pop) {
            mean_fitness += expr.fitness / (double)pop.size();
            mean_length += expr.num_of_nodes / (double)pop.size();
        }

        // Find the covariance of fitness and number of nodes
        double covariance = 0.0;
        for (const auto &expr : pop) {
            covariance += (expr.fitness - mean_fitness) * (expr.num_of_nodes - mean_length) / (double)pop.size();
        }

        // Find the variance of number of nodes
        double variance_length = 0.0;
        for (const auto &expr : pop) {
            variance_length += (expr.num_of_nodes - mean_length) * (expr.num_of_nodes - mean_length) / (double)pop.size();
        }

        // Find the parsimony pressure coefficient
        double ct = covariance / variance_length;

        // Subtract from the fitness of each function the parsimony pressure term
        for (auto &expr : pop) {
            expr.fitness -= ct * expr.num_of_nodes;
        }
    }
}

LearningHistory GeneticProgramming::fit(std::shared_ptr<const Dataset> dataset, int ngenerations, int nepochs, double learning_rate) noexcept {
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

    int elitism_count = ceil(config.elite_rate * config.npopulation);

    std::vector<const Expression *> parents1;
    std::vector<const Expression *> parents2;
    parents1.resize(config.npopulation);
    parents2.resize(config.npopulation);

    std::vector<Expression> children1;
    std::vector<Expression> children2;
    children1.resize(config.npopulation);
    children2.resize(config.npopulation);

    // Iterate for ngenerations
    for (int generation = 0; generation < ngenerations; ++generation)
    {
        // Sort population by fitness in descending order
        std::sort(population.begin(), population.end(), std::greater<Expression>());

        // Update selection probabilities
        selector->update(&population[0]);

        /* Copy the elite directly to offspring */
        int counter = elitism_count;

        /* Generate the rest from parents */
        omp_set_max_active_levels(2);
        int nested_thread_count = std::min(omp_get_num_procs() / nislands - 1, 1);
#pragma omp parallel for num_threads(nested_thread_count) schedule(static)
        for (int i = 0; i < config.npopulation; ++i) {
            parents1[i] = &selector->select(&population[0]);
        }
#pragma omp parallel for num_threads(nested_thread_count) schedule(static)
        for (int i = 0; i < config.npopulation; ++i) {
            parents2[i] = &selector->select(&population[0]);
        }
#pragma omp parallel for num_threads(nested_thread_count) schedule(dynamic)
        for (int i = 0; i < config.npopulation; ++i) {
            const auto &c12 = recombiner->recombine(*parents1[i], *parents2[i]);
            children1[i] = std::move(get<0>(c12));
            children2[i] = std::move(get<1>(c12));
        }
#pragma omp parallel for num_threads(nested_thread_count) schedule(dynamic)
        for (int i = 0; i < config.npopulation; ++i) {
            children1[i] = std::move(mutator->mutate(children1[i]));
        }
#pragma omp parallel for num_threads(nested_thread_count) schedule(dynamic)
        for (int i = 0; i < config.npopulation; ++i) {
            children2[i] = std::move(mutator->mutate(children2[i]));
        }
        for (int i = 0; i < config.npopulation && counter < config.npopulation; ++i) {
            population[counter++] = std::move(children1[i]);
            if (counter < config.npopulation)
                population[counter++] = std::move(children2[i]);
        }

        // Compute losses
        runner->run(population, dataset, nepochs, learning_rate);

        // Calculate fitnesses
        calculate_fitnesses(population, config.enable_parsimony_pressure);

        // Find new best
        const Expression best = *get_best_solution();

        // Append to learning history
        history.add_to_history(best);
    }

    return history;
}

}