// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "hoist_mutator.hpp"

#include "util/rng.hpp"

namespace qsr {
    Expression HoistMutator::mutate(const Expression &expr) noexcept {
        if (((thread_local_rng() % RAND_MAX) / (float)RAND_MAX) > mutation_probability) {
            return expr;
        }

        // Make a copy of the original expression
        Expression result = expr;

        // Pick a random subtree
        Expression &random_subtree = expression_picker.pick(result);

        // Pick a random node within the subtree
        const Expression random_node = expression_picker.pick(random_subtree);

        // Replace the random subtree with the random node
        random_subtree = random_node;

        // In case the selected random subtree is the root, get the weights from the original expression
        result.weights = expr.weights;

        // Return the mutated expression
        return expression_reorganizer.reorganize(result);
    }
}