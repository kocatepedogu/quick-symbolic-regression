#include "fitness_proportional_selector.hpp"

#include <random>

void FitnessProportionalSelector::update(const Expression population[]) {
    // Step 1: Calculate the total fitness
    float totalFitness = 0.0f;
    for (size_t i = 0; i < npopulation; ++i) {
        // Convert loss to fitness (reciprocal)
        float fitness = 1.0f / population[i].loss;
        totalFitness += fitness;
    }

    // Step 2: Normalize fitness values to get probabilities
    for (size_t i = 0; i < npopulation; ++i) {
        float fitness = 1.0f / population[i].loss;
        probabilities[i] = fitness / totalFitness;
    }
}


const Expression& FitnessProportionalSelector::select(const Expression population[]) {
    // Create a random number generator
    static thread_local std::random_device rd;
    static thread_local std::mt19937 gen(rd());
    
    std::discrete_distribution<> distribution(
        probabilities.begin(), probabilities.end());

    // Select an expression based on the probabilities
    size_t selectedIndex = distribution(gen);
    return population[selectedIndex];
}