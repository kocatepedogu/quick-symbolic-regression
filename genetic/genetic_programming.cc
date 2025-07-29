#include "genetic_programming.hpp"

#include "../expressions/expression.hpp"

#include "initializer/base.hpp"
#include "learning_history.hpp"

#include <iostream>

GeneticProgramming::GeneticProgramming(
               std::shared_ptr<const Dataset> dataset, 
               int nweights, 
               int npopulation, 
               std::shared_ptr<BaseInitializer> initializer,
               std::shared_ptr<BaseMutation> mutator,
               std::shared_ptr<BaseCrossover> crossover,
               std::shared_ptr<BaseSelection> selection,
               std::shared_ptr<BaseRunner> runner) noexcept : 
               dataset(dataset), 
               nvars(dataset->n), 
               nweights(nweights), 
               npopulation(npopulation % 2 == 0 ? npopulation : npopulation + 1), 
               initializer(initializer),
               mutator(mutator),
               crossover(crossover),
               selection(selection),
               runner(runner)
{
    // Initialize island with a population of random expressions
    initializer->initialize(population);

    // Get selector
    selector = selection->get_selector(npopulation);
}

Expression GeneticProgramming::get_best_solution() {
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

void GeneticProgramming::insert_solution(Expression e) {
    population[npopulation - 2] = e;
}

LearningHistory GeneticProgramming::fit(int ngenerations, int nepochs, float learning_rate) noexcept {
    // Create empty learning history
    LearningHistory history;

    // Compute initial fitnesses
    runner->run(population, nepochs, learning_rate);

    // Iterate for ngenerations
    for (int generation = 0; generation < ngenerations; ++generation)
    {
        Expression best = get_best_solution();

        selector->update(&population[0]);

        /* Offspring Generation */
        std::vector<Expression> offspring;
        for (int i = 0; i < npopulation / 2; ++i) {
            const auto &parent1 = selector->select(&population[0]);
            const auto &parent2 = selector->select(&population[0]);
            const auto &children = crossover->crossover(parent1, parent2);
            const auto &child1 = get<0>(children);
            const auto &child2 = get<1>(children);
            offspring.push_back(mutator->mutate(child1));
            offspring.push_back(mutator->mutate(child2));
        }

        // Replace current population with offspring
        population = offspring;

        // Compute fitnesses
        runner->run(population, nepochs, learning_rate);

        // Preserve previous best
        population[0] = best;

        // Append to learning history
        history.add_to_history(best);
    }

    return history;
}
