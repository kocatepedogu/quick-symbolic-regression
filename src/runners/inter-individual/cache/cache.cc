// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "../../../../include/quicksr/runners/inter-individual/cache/cache.hpp"

#include <iostream>

namespace qsr::inter_individual {

void Cache::read_from_population(const std::vector<Expression>& population) {
    uncached_population = std::vector<Expression>();
    uncached_indices = std::vector<int>();

    cached_population = std::vector<Expression>();
    cached_indices = std::vector<int>();

    for (int i = 0; i < population.size(); ++i) {
        const auto& expr = population[i];

        if (cache.contains(expr) && *cache.find(expr) == expr) {
            cached_population.push_back(*cache.find(expr));
            cached_indices.push_back(i);
        } else {
            uncached_population.push_back(expr);
            uncached_indices.push_back(i);
        }
    }
}

void Cache::write_to_population(std::vector<Expression>& population) const {
    for (int i = 0; i < cached_population.size(); ++i) {
        population[cached_indices[i]] = cached_population[i];
    }

    for (int i = 0; i < uncached_population.size(); ++i) {
        population[uncached_indices[i]] = uncached_population[i];
    }
}

void Cache::save() {
    for (const Expression& expr : uncached_population) {
        cache.insert(expr);
    }
}

}