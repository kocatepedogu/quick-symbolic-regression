// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef EXPRESSION_GENERATOR_FULL_HPP
#define EXPRESSION_GENERATOR_FULL_HPP

#include "expressions/expression.hpp"
#include "config.hpp"
#include "genetic/common/function_set.hpp"

#include <memory>
#include <random>

namespace qsr {

/**
 * @brief Generates expressions with a symmetric and full parse tree whose depth is equal to config.max_depth
 */
class FullExpressionGenerator {
public:
    FullExpressionGenerator(const Config &config);

    /**
     * @brief Generates a random expression whose parse tree is full and has depth equal to config.max_depth
     */
    Expression generate() noexcept;

private:

    /**
     * @brief Generates a random expression whose parse tree is full and has depth equal to remaining_depth argument
     *
     * @param remaining_depth Exact depth of the parse tree representing the generated expression
     *
     * @details
     * The method recursively calls itself, with a decrement in the max_depth argument at every call.
     *
     * If `remaining_depth` >= 2, it only generates unary/binary operations at recursive invocations.
     * Terminals/leaf nodes are never generated unless `remaining_depth` is one, meaning that the tree is
     * full and symmetric with no early termination at any branch.
     *
     * If `remaining_depth` == 1, it only generates terminals (varibles or trainable parameters)
     */
    Expression generate(int remaining_depth) noexcept;

    /**
     * @brief Probability distribution of functions to be selected when remaining_depth == 1 in generate method
     */
    std::discrete_distribution<> depth_one_distribution;

    /**
     * @brief Probability distribution of functions to be selected when remaining_depth >= 2 in generate method
     */
    std::discrete_distribution<> depth_two_distribution;

    Config config;
};

}

#endif