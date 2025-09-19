// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "genetic/mutation/mutator/distribution_mutator.hpp"
#include "util/rng.hpp"

#include <random>

namespace qsr {
    Expression DistributionMutator::mutate(const Expression &expr) noexcept {
        std::discrete_distribution<> distribution(
            probabilities.begin(), probabilities.end());

        // Select a mutator based on the probabilities
        size_t selectedIndex = distribution(thread_local_rng);
        
        // Mutate and return the expression using the selected mutator
        return mutators[selectedIndex]->mutate(expr);
    }
}