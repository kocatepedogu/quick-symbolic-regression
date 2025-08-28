// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef EXPRESSION_GENERATOR_HPP
#define EXPRESSION_GENERATOR_HPP

#include "expressions/expression.hpp"
#include "config.hpp"

#include <random>

namespace qsr {

/**
 * @brief Generates expressions with varying structure and depth
 */
class ExpressionGenerator {
public:
    ExpressionGenerator();
    ExpressionGenerator(const Config &config);

    /**
     * @brief Generates a random expression whose parse tree depth is limited by config.max_depth
     */
    Expression generate() noexcept;

private:
    /**
     * @brief Generates a random expression whose parse tree depth is limited by the max_depth argument
     *
     * @param max_depth Maximum depth of generated expressions
     *
     * @details
     * The method recursively calls itself, with a decrement in the max_depth argument at every call.
     *
     * As long as `max_depth` >= 2, it is allowed to generate both terminals and unary/binary operations
     * at recursive invocations, meaning that different branches of the generated parse tree can have
     * different depth and structure due to early termination in some branches.
     *
     * If `max_depth` == 1, it only generates terminals (variables or trainable parameters)
     */
    Expression generate(int max_depth) noexcept;

    /**
     * @brief Generates a random ordinal corresponding to an enum value in operation_t
     */
    int random_operation(int max_depth) noexcept;

    /**
     * @brief Probability distribution of functions to be selected when `max_depth == 1` in generate method
     */
    std::discrete_distribution<> depth_one_distribution;

    /**
     * @brief Probability distribution of functions to be selected when `max_depth >= 2` in generate method
     */
    std::discrete_distribution<> depth_two_distribution;

    Config config;
};
}

#endif