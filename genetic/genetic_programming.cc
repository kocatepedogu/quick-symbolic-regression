#include "genetic_programming.hpp"

#include "../expressions/expression.hpp"
#include "../util/rng.hpp"

#include "expression_comparator.hpp"
#include "expression_generator.hpp"

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
               crossover(crossover_probability),
               probabilities(npopulation)
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
    int max_index = -1;
    float max_fitness = -1e30;
    for (int i = 0; i < npopulation; ++i) {
        float fitness = -population[i].loss;
        if (fitness > max_fitness) {
            max_fitness = fitness;
            max_index = i;
        }
    }

    return population[max_index];
}

void Island::insert_solution(Expression e) {
    population[npopulation - 2] = e;
}

void Island::parent_selection_fitness_proportional_probs() noexcept {
    float min_fitness = 1e30;
    for (int i = 0; i < npopulation; ++i) {
        float fitness = -population[i].loss;
        if (fitness < min_fitness) {
            min_fitness = fitness;
        }
    }

    float sum_of_fitnesses = 0;
    for (int i = 0; i < npopulation; ++i) {
        float fitness = (-population[i].loss) - min_fitness;
        sum_of_fitnesses += fitness;
    }

    // Compute selection probabilities for each individual
    for (int i = 0; i < npopulation; ++i) {
        probabilities[i] = ((-population[i].loss) - min_fitness) / sum_of_fitnesses;
    }
}

int Island::parent_selection_fitness_proportional() const noexcept {
    float u = (thread_local_rng() % RAND_MAX) / (float)RAND_MAX;
    float s = 0;

    int parent_index = 0;
    for (; parent_index < npopulation && u > s; ++parent_index) {
        s += probabilities[parent_index];
    }

    if (parent_index < 0) parent_index = 0;
    if (parent_index > npopulation - 1) parent_index = npopulation - 1;

    return parent_index;
}

void Island::iterate(int niters) noexcept {
    for (int iter = 0; iter < niters; ++iter)
    {
        parent_selection_fitness_proportional_probs();
        Expression best = get_best_solution();

        /* Offspring Generation */
        std::vector<Expression> offspring;
        for (int i = 0; i < npopulation / 2; ++i) {
            const auto &parent1 = parent_selection_fitness_proportional();
            const auto &parent2 = parent_selection_fitness_proportional();
            const auto &children = crossover.crossover(parent1, parent2);
            const auto &child1 = get<0>(children);
            const auto &child2 = get<1>(children);
            offspring.push_back(mutator.mutate(child1));
            offspring.push_back(mutator.mutate(child2));
        }

        // Replace current population with offspring
        population = offspring;

        // Compute fitnesses
        runner.run(population, 2);

        // Preserve previous best
        population[0] = best;
    }
}