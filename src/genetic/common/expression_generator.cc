// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "genetic/common/expression_generator.hpp"

#include "util/rng.hpp"
#include "util/macro.hpp"
#include "expressions/unary.hpp"
#include "expressions/binary.hpp"

#include <cassert>
#include <random>

namespace qsr {
ExpressionGenerator::ExpressionGenerator()
    : generate_full_trees(false) {}

ExpressionGenerator::ExpressionGenerator(const Config &config, bool generate_full_trees) :
    config(config), generate_full_trees(generate_full_trees) {}

Expression ExpressionGenerator::generate(int max_depth, int current_depth) noexcept {
    #define _RANDOM_EXPR_CALL(i) \
        generate(max_depth - 1, current_depth + 1)

    auto terminal_or_function_distribution = std::discrete_distribution<>({
        // Terminal
        1.0,
        // Function
        2.0
    });

    int terminal_or_function = terminal_or_function_distribution(thread_local_rng);
    if (terminal_or_function == 0 && !generate_full_trees || max_depth == 1) {
        // Choose a terminal operation
        auto distribution = std::discrete_distribution<>({
            1.0,
            1.0,
            config.nweights > 0 ? 1.0 : 0.0,
       });

        switch (distribution(thread_local_rng)) {
        case 0:
            return 2 * (thread_local_rng() % RAND_MAX) / (double)RAND_MAX - 1;
        case 1:
            return Var(thread_local_rng() % config.nvars);
        case 2:
            return Parameter(thread_local_rng() % config.nweights);
        default:break;
        }
    }
    else {
        auto binary_or_unary_distribution = std::discrete_distribution<>({
            // Binary
            0.5,
            // Unary
            0.5
        });

        int binary_or_unary = binary_or_unary_distribution(thread_local_rng);
        if (binary_or_unary == 0) {
            // Choose a binary operation
            auto distribution = std::discrete_distribution<>({
               config.function_set->addition ? 1.0 : 0.0,
               config.function_set->subtraction ? 1.0 : 0.0,
               config.function_set->multiplication ? 1.0 : 0.0,
               config.function_set->division ? 1.0 : 0.0,
           });

            switch (distribution(thread_local_rng)) {
                BINARY_OP_CASE(0, _RANDOM_EXPR_CALL, +);
                BINARY_OP_CASE(1, _RANDOM_EXPR_CALL, -);
                BINARY_OP_CASE(2, _RANDOM_EXPR_CALL, *);
                BINARY_OP_CASE(3, _RANDOM_EXPR_CALL, /);
            default:break;
            }
        }
        else {
            // Choose a unary operation
            auto distribution = std::discrete_distribution<>({
               config.function_set->sine ? 1.0 : 0.0,
               config.function_set->cosine ? 1.0 : 0.0,
               config.function_set->exponential ? 1.0 : 0.0,
               config.function_set->rectified_linear_unit ? 1.0 : 0.0,
           });

            switch (distribution(thread_local_rng)) {
                UNARY_OP_CASE(0, _RANDOM_EXPR_CALL, Sin);
                UNARY_OP_CASE(1, _RANDOM_EXPR_CALL, Cos);
                UNARY_OP_CASE(2, _RANDOM_EXPR_CALL, Exp);
                UNARY_OP_CASE(3, _RANDOM_EXPR_CALL, ReLU);
            default:break;
            }
        }
    }

    // Control must never reach here.
    fprintf(stderr, "ExpressionGenerator::generate Illegal state encountered.\n");
    abort();
}

Expression ExpressionGenerator::generate() noexcept {
    return generate(config.max_depth, 1);
}

}