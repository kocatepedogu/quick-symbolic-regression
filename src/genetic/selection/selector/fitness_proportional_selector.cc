// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "genetic/selection/selector/fitness_proportional_selector.hpp"
#include "util/rng.hpp"

#include <random>

namespace qsr {

void FitnessProportionalSelector::update(const Expression population[]) {
    // Step 1: Calculate the total fitness
    float total_fitness = 0.0f;
    #pragma omp simd reduction(+:total_fitness)
    for (size_t i = 0; i < npopulation; ++i) {
        total_fitness += population[i].fitness;
    }

    // Step 2: Normalize fitness values to get probabilities
    #pragma omp simd
    for (size_t i = 0; i < npopulation; ++i) {
        probabilities[i] = population[i].fitness / total_fitness;
    }
}


const Expression& FitnessProportionalSelector::select(const Expression population[]) {
    // Create a random number generator
    std::discrete_distribution<> distribution(
        probabilities.begin(), probabilities.end());

    // Select an expression based on the probabilities
    size_t selectedIndex = distribution(thread_local_rng);
    return population[selectedIndex];
}

}