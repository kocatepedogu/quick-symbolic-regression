// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "genetic/initialization/initializer/ramped_half_and_half_initializer.hpp"
#include "util/rng.hpp"
#include <algorithm> // for std::shuffle
#include <vector>

namespace qsr {

RampedHalfAndHalfInitializer::RampedHalfAndHalfInitializer(const Config &config) :
    config(config),
    generator(config) // Initialize the single generator instance here
{}

void RampedHalfAndHalfInitializer::initialize(std::vector<Expression>& population) {
    population.clear();
    population.reserve(config.npopulation);

    const int min_depth = 2; // A reasonable minimum depth for non-trivial trees.
    const int max_init_depth = config.max_depth;

    if (max_init_depth < min_depth) {
        // Fallback: if max_depth is too small, just use grow method at the max depth.
        for (int i = 0; i < config.npopulation; ++i) {
            population.push_back(generator.generate(max_init_depth, false));
        }
        return;
    }

    const int num_depth_levels = max_init_depth - min_depth + 1;
    const int individuals_per_level = config.npopulation / num_depth_levels;
    int remainder = config.npopulation % num_depth_levels;

    for (int d = min_depth; d <= max_init_depth; ++d) {
        int count_for_this_depth = individuals_per_level;
        if (d - min_depth < remainder) {
            count_for_this_depth++;
        }

        if (count_for_this_depth == 0) continue;

        int half_size = count_for_this_depth / 2;
        int remainder_half = count_for_this_depth % 2;

        // Generate 'grow' trees for this depth
        for (int i = 0; i < half_size; ++i) {
            population.push_back(generator.generate(d, false));
        }

        // Generate 'full' trees for this depth
        for (int i = 0; i < half_size + remainder_half; ++i) {
            population.push_back(generator.generate(d, true));
        }
    }

    while (population.size() < config.npopulation) {
        population.push_back(generator.generate(max_init_depth, false));
    }
    if (population.size() > config.npopulation) {
        population.resize(config.npopulation);
    }

    // Shuffle the final population to mix trees of different depths and types
    auto& rng = thread_local_rng;
    std::shuffle(population.begin(), population.end(), rng);
}

}
