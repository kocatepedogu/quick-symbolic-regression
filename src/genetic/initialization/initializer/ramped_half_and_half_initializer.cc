// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "genetic/initialization/initializer/ramped_half_and_half_initializer.hpp"
#include "util/rng.hpp"
#include <algorithm> // for std::shuffle
#include <vector>
#include <omp.h> // For OpenMP

namespace qsr {
extern int nislands;

RampedHalfAndHalfInitializer::RampedHalfAndHalfInitializer(const Config &config) :
    config(config),
    generator(config) // Initialize the single generator instance here
{}

void RampedHalfAndHalfInitializer::initialize(std::vector<Expression>& population) {
    population.clear();
    population.reserve(config.npopulation);

    const int num_threads = std::min(omp_get_num_procs() / nislands - 1, 1);
    std::vector<std::vector<Expression>> thread_populations(num_threads);

    // Calculate population size for each thread
    std::vector<int> pop_per_thread(num_threads);
    int base_pop_size = config.npopulation / num_threads;
    int remainder_pop = config.npopulation % num_threads;
    for(int i = 0; i < num_threads; ++i) {
        pop_per_thread[i] = base_pop_size + (i < remainder_pop ? 1 : 0);
    }

    omp_set_max_active_levels(2);
    #pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();
        int local_pop_size = pop_per_thread[thread_id];
        auto& local_population = thread_populations[thread_id];
        local_population.reserve(local_pop_size);

        // Each thread gets its own instance of the generator
        ExpressionGenerator local_generator(config);

        const int min_depth = 1;
        const int max_init_depth = config.max_depth;

        if (max_init_depth < min_depth) {
            for (int i = 0; i < local_pop_size; ++i) {
                local_population.push_back(local_generator.generate(max_init_depth, false));
            }
        } else {
            const int num_depth_levels = max_init_depth - min_depth + 1;
            const int individuals_per_level = local_pop_size / num_depth_levels;
            int remainder = local_pop_size % num_depth_levels;

            for (int d = min_depth; d <= max_init_depth; ++d) {
                int count_for_this_depth = individuals_per_level;
                if (d - min_depth < remainder) {
                    count_for_this_depth++;
                }
                if (count_for_this_depth == 0) continue;

                int half_size = count_for_this_depth / 2;
                int remainder_half = count_for_this_depth % 2;

                for (int i = 0; i < half_size; ++i) {
                    local_population.push_back(local_generator.generate(d, false));
                }
                for (int i = 0; i < half_size + remainder_half; ++i) {
                    local_population.push_back(local_generator.generate(d, true));
                }
            }
             // Fill any remaining spots due to rounding
            while (local_population.size() < local_pop_size) {
                local_population.push_back(local_generator.generate(max_init_depth, false));
            }
        }
    } // End of parallel region

    // Merge results from all threads
    for (int i = 0; i < num_threads; ++i) {
        population.insert(population.end(), thread_populations[i].begin(), thread_populations[i].end());
    }

    // Ensure final population is exactly the required size and shuffle
    if (population.size() > config.npopulation) {
        population.resize(config.npopulation);
    }

    auto& rng = thread_local_rng;
    std::shuffle(population.begin(), population.end(), rng);
}

} // namespace qsr

