// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "rank_selector.hpp"

#include <algorithm>
#include <random>

namespace qsr {
    void RankSelector::update(const Expression population[]) {
        // Fill the vector of indices
        for (int i = 0; i < npopulation; i++) {
            indices[i] = i;
        }

        // Sort indices based on population fitness in descending order
        std::stable_sort(indices.begin(), indices.end(),
              [&population](int i, int j) {
                  return population[i].loss < population[j].loss;
        });

        // Calculate probabilities based on rank
        for (int i = 0; i < npopulation; i++) {
            probabilities[i] = (1.0 / npopulation) * (1.0 + sp * (1 - 2.0 * indices[i] / (npopulation - 1.0)));
        }
    }

    const Expression& RankSelector::select(const Expression population[]) {
        // Create a random number generator
        static thread_local std::random_device rd;
        static thread_local std::mt19937 gen(rd());
        
        std::discrete_distribution<> distribution(
            probabilities.begin(), probabilities.end());

        // Select an expression based on the probabilities
        size_t selectedIndex = distribution(gen);
        return population[selectedIndex];
    }
}