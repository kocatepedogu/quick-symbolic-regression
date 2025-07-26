#include "fitness_proportional_selection.hpp"

#include "../../util/rng.hpp"

void FitnessProportionalSelection::initialize(const Expression population[]) {
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

const Expression& FitnessProportionalSelection::select(const Expression population[]) {
    float u = (thread_local_rng() % RAND_MAX) / (float)RAND_MAX;
    float s = 0;

    int parent_index = 0;
    for (; parent_index < npopulation && u > s; ++parent_index) {
        s += probabilities[parent_index];
    }

    if (parent_index < 0) parent_index = 0;
    if (parent_index > npopulation - 1) parent_index = npopulation - 1;

    return population[parent_index];
}