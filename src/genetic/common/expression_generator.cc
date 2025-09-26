// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "genetic/common/expression_generator.hpp"

#include "util/rng.hpp"
#include "util/macro.hpp"
#include "expressions/unary.hpp"
#include "expressions/binary.hpp"

#include <cassert>
#include <random>
#include <vector>
#include <numeric>

namespace qsr {

ExpressionGenerator::ExpressionGenerator(const Config &config) :
    config(config)
{
    // --- Pre-calculate distributions for performance ---

    // Terminal choices: Heavily bias towards variables and simple constants.
    std::vector<double> terminal_weights = {
        2.0, // Constant
        4.0, // Variable
        config.nweights > 0 ? 1.5 : 0.0 // Parameter
    };
    m_terminal_dist = std::discrete_distribution<>(terminal_weights.begin(), terminal_weights.end());

    // Unary operator choices
    const auto& fs = *config.function_set;
    std::vector<double> unary_weights = {
        fs.sine ? 1.0 : 0.0, fs.cosine ? 1.0 : 0.0,
        fs.exponential ? 1.0 : 0.0, fs.rectified_linear_unit ? 1.0 : 0.0
    };
    if (std::accumulate(unary_weights.begin(), unary_weights.end(), 0.0) > 1e-6) {
        m_unary_dist = std::discrete_distribution<>(unary_weights.begin(), unary_weights.end());
        m_has_unary = true;
    }

    // Binary operator choices
    std::vector<double> binary_weights = {
        fs.addition ? 1.0 : 0.0, fs.subtraction ? 1.0 : 0.0,
        fs.multiplication ? 1.0 : 0.0, fs.division ? 1.0 : 0.0
    };
    if (std::accumulate(binary_weights.begin(), binary_weights.end(), 0.0) > 1e-6) {
        m_binary_dist = std::discrete_distribution<>(binary_weights.begin(), binary_weights.end());
        m_has_binary = true;
    }

    if (!m_has_unary && !m_has_binary) {
        // This should be an exception in a real application
        fprintf(stderr, "ExpressionGenerator Error: No functions available in the function set.\n");
        abort();
    }

    // Distribution for choosing between Unary and Binary functions
    std::vector<double> func_type_weights = { m_has_binary ? 1.0 : 0.0, m_has_unary ? 1.0 : 0.0 };
    m_func_type_dist = std::discrete_distribution<>(func_type_weights.begin(), func_type_weights.end());

    // General-purpose distributions
    m_uniform_dist = std::uniform_real_distribution<>(-1.0, 1.0);
    if (config.nvars > 0) {
        m_var_dist = std::uniform_int_distribution<>(0, config.nvars - 1);
    }
    if (config.nweights > 0) {
        m_param_dist = std::uniform_int_distribution<>(0, config.nweights - 1);
    }
}

Expression ExpressionGenerator::generate(int max_depth, bool generate_full_trees) noexcept {
    return generate_recursive(max_depth, 1, generate_full_trees);
}

Expression ExpressionGenerator::generate_recursive(int max_depth, int current_depth, bool generate_full_trees) noexcept {
    bool is_terminal_depth = (current_depth >= max_depth);

    // PTC1: Probability of choosing a non-terminal.
    // A common value for the coefficient is 0.9.
    double prob_non_terminal = 0.9;

    std::bernoulli_distribution non_terminal_dist(prob_non_terminal);

    bool create_function_node = !is_terminal_depth && (generate_full_trees || non_terminal_dist(thread_local_rng));

    if (!create_function_node) {
        // --- Generate a Terminal Node ---
        switch (m_terminal_dist(thread_local_rng)) {
            case 0: // Constant
            {
                // Heavily bias toward simple, useful integers.
                std::discrete_distribution<> const_dist({
                    40.0, // P(-1.0)
                    40.0, // P(1.0)
                    15.0, // P(0.0)
                     5.0  // P(random float)
                });
                switch(const_dist(thread_local_rng)) {
                    case 0: return {-1.0};
                    case 1: return {1.0};
                    case 2: return {0.0};
                    default: return {m_uniform_dist(thread_local_rng)};
                }
            }
            case 1: // Variable
                 if (config.nvars > 0) {
                    return Var(m_var_dist(thread_local_rng));
                }
                // Fallback to a constant if no variables are configured
                return {1.0};
            case 2: // Parameter
                if (config.nweights > 0) {
                    return Parameter(m_param_dist(thread_local_rng));
                }
                // Fallback to a constant if no parameters are configured
                return {0.0};
        }
    } else {
        // --- Generate a Function Node ---
        enum { BINARY, UNARY };
        int func_type = m_has_unary && !m_has_binary ? UNARY :
                        !m_has_unary && m_has_binary ? BINARY :
                        m_func_type_dist(thread_local_rng);

        if (func_type == BINARY) {
            Expression e1 = generate_recursive(max_depth, current_depth + 1, generate_full_trees);
            Expression e2 = generate_recursive(max_depth, current_depth + 1, generate_full_trees);
            switch (m_binary_dist(thread_local_rng)) {
                case 0: return e1 + e2;
                case 1: return e1 - e2;
                case 2: return e1 * e2;
                case 3: return e1 / e2;
            }
        } else { // UNARY
            Expression e = generate_recursive(max_depth, current_depth + 1, generate_full_trees);
            switch (m_unary_dist(thread_local_rng)) {
                case 0: return Sin(e);
                case 1: return Cos(e);
                case 2: return Exp(e);
                case 3: return ReLU(e);
            }
        }
    }

    // Fallback, should be unreachable
    fprintf(stderr, "ExpressionGenerator::generate_recursive reached an illegal state.\n");
    abort();
}

} // namespace qsr

