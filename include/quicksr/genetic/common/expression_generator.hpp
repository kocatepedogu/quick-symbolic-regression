// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef EXPRESSION_GENERATOR_HPP
#define EXPRESSION_GENERATOR_HPP

#include "expressions/expression.hpp"
#include "config.hpp"

#include <random>
#include <vector>

namespace qsr {

/**
 * @brief Generates expressions using the PTC1 (Probabilistic Tree Creation 1) algorithm.
 *
 * @details This generator is designed to produce a diverse and high-quality initial
 * population. It incorporates biased constant generation, favoring simple integers
 * like -1, 0, and 1, which are often more useful in symbolic regression than
 * random floating-point numbers. The PTC1 algorithm provides a structured way
 * to control tree growth, ensuring a good distribution of tree sizes and shapes.
 */
class ExpressionGenerator {
public:
    explicit ExpressionGenerator(const Config &config);

    /**
     * @brief Generates a random expression.
     * @param max_depth The maximum depth for this specific tree.
     * @param generate_full_trees If true, generates trees where every branch reaches the maximum depth.
     * If false, uses the "grow" method, allowing for varied tree shapes.
     */
    Expression generate(int max_depth, bool generate_full_trees) noexcept;

private:
    Expression generate_recursive(int max_depth, int current_depth, bool generate_full_trees) noexcept;

    const Config config;

    // --- Pre-computed distributions for performance ---
    std::discrete_distribution<> m_node_selection_dist;
    std::discrete_distribution<> m_terminal_dist;
    std::discrete_distribution<> m_unary_dist;
    std::discrete_distribution<> m_binary_dist;
    std::discrete_distribution<> m_func_type_dist;
    std::uniform_real_distribution<double> m_uniform_dist;
    std::uniform_int_distribution<> m_var_dist;
    std::uniform_int_distribution<> m_param_dist;

    bool m_has_unary = false;
    bool m_has_binary = false;


};
}

#endif

