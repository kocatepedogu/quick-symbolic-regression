// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "distribution_mutator.hpp"

#include <random>

namespace qsr {
    Expression DistributionMutator::mutate(const Expression &expr) noexcept {
        // Create a random number generator
        static thread_local std::random_device rd;
        static thread_local std::mt19937 gen(rd());
        
        std::discrete_distribution<> distribution(
            probabilities.begin(), probabilities.end());

        // Select a mutator based on the probabilities
        size_t selectedIndex = distribution(gen);
        
        // Mutate and return the expression using the selected mutator
        return mutators[selectedIndex]->mutate(expr);
    }
}