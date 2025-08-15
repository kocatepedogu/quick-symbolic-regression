// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "subtree_mutator.hpp"

namespace qsr {

Expression SubtreeMutator::mutate(const Expression &expr) noexcept {
    const auto &random_expr = expression_generator.generate();
    const auto &result_pair = recombiner.recombine(expr, random_expr);

    Expression result = 0.0;

    // Substitute random subtree into the original expression (50% chance)
    if (rand() % 2 == 0) {
        result = get<0>(result_pair);
    }
    // Substitute original expression into the random subtree (50% chance)
    else {
        result = get<1>(result_pair);
    }

    // If the number of nodes exceed the maximum depth, revert to original expression
    if (result.num_of_nodes > max_depth) {
        result = expr;
    }

    // Use the same weights as the original expression
    result.weights = expr.weights;

    // Return the mutated expression
    return result;
}

}