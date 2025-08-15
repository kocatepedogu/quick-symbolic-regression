// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "default.hpp"

#include "../../../util/rng.hpp"

#include "../../expression_picker.hpp"

#include <tuple>
#include <cassert>

namespace qsr {

std::tuple<Expression, Expression> DefaultRecombiner::recombine(Expression e1, Expression e2) noexcept {
    if (((thread_local_rng() % RAND_MAX) / (float)RAND_MAX) > crossover_probability) {
        return std::make_tuple(e1, e2);
    }

    // Save original expressions
    const Expression e1_original(e1);
    const Expression e2_original(e2);

    // Get original weights from both expressions

    const std::vector<float> weights1 = e1.weights;
    const std::vector<float> weights2 = e2.weights;

    // Exchange subtrees

    Expression &sub1 = expression_picker.pick(e1);
    Expression copy_sub_1(sub1);

    Expression &sub2 = expression_picker.pick(e2);
    Expression copy_sub_2(sub2);

    sub1 = copy_sub_2;
    sub2 = copy_sub_1;

    // Update number of nodes in each tree and apply optimizations

    Expression e1_reorg = expression_reorganizer.reorganize(e1);
    Expression e2_reorg = expression_reorganizer.reorganize(e2);

    // If the number of nodes exceed the maximum depth, revert to original trees
    if (e1.num_of_nodes > max_depth) {
        e1_reorg = e1_original;
    }
    if (e2.num_of_nodes > max_depth) {
        e2_reorg = e2_original;
    }

    // If both trees have weights, apply whole arithmetic crossover to weights

    if (!weights1.empty() && !weights2.empty()) {
        assert(weights1.size() == weights2.size());

        e1_reorg.weights.resize(weights1.size());
        e2_reorg.weights.resize(weights2.size());

        for (int i = 0; i < weights1.size(); i++) {
            float alpha = (thread_local_rng() % RAND_MAX) / (float)RAND_MAX;

            float w1 = alpha * weights1[i] + (1 - alpha) * weights2[i];
            float w2 = alpha * weights2[i] + (1 - alpha) * weights1[i];

            e1_reorg.weights[i] = w1;
            e2_reorg.weights[i] = w2;
        }
    }

    // If one tree has weights and the other does not, copy the weights from
    // the tree which has weights

    if (!weights1.empty() && weights2.empty()) {
        e1_reorg.weights = weights1;
        e2_reorg.weights = weights1;
    }
    
    if (weights1.empty() && !weights2.empty()) {
        e1_reorg.weights = weights2;
        e2_reorg.weights = weights2;
    }

    // Make a pair of expressions (may become parents)

    return std::make_tuple(e1_reorg, e2_reorg);
}

}