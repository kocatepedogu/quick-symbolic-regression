// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef CONFIG_HPP
#define CONFIG_HPP

#include "function_set.hpp"
#include <memory>

namespace qsr {
    /**
     * @brief Stores common configuration options required by many genetic operators and algoritms
     */
    struct Config {
        Config();
        
        /**
         * @brief All arguments constructor
         */
        Config(int nvars, int nweights, int max_depth, int npopulation, float elite_rate, float migration_rate, std::shared_ptr<FunctionSet> function_set);

        /**
         * @brief Number of variables (features) in the training data
         */
        int nvars;

        /**
         * @brief Maximum number of trainable parameters (weights) allowed in an expression
         */
        int nweights;

        /**
         * @brief The maximum depth expression parse trees are allowed to reach
         */
        int max_depth;

        /**
         * @brief The number of expressions in a single generation
         * @details
         * This is the population size commmonly denoted by the Greek letter
         * mu in the literature.
         */
        int npopulation;

        /**
         * @brief Proportion of elite individuals that will be copied directly to the next generation
         */
        float elite_rate;

        /**
         * @brief Proportion of individuals that migrate between islands at the end of every supergeneration
         */
        float migration_rate;

        /**
         * @brief The function_set stores the set of unary and binary functions that can be present in the evolving functions.
         */
        std::shared_ptr<FunctionSet> function_set;
    };
}

#endif // CONFIG_HPP
