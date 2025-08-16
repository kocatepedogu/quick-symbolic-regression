// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "subtree_mutator.hpp"

#include "../../../util/rng.hpp"

namespace qsr {

SubtreeMutator::SubtreeMutator(const Config &config, float mutation_probability, int max_depth_increment)  :
    config(config),
    mutation_probability(mutation_probability),
    max_depth_increment(max_depth_increment)
{
    Config expression_generator_config(config);
    expression_generator_config.max_depth = max_depth_increment;
    expression_generator = ExpressionGenerator(expression_generator_config);
}

    
Expression SubtreeMutator::mutate(const Expression &expr) noexcept {
    if (((thread_local_rng() % RAND_MAX) / (float)RAND_MAX) > mutation_probability) {
        return expr;
    }

    // Make a copy of the original expression
    Expression result = expr;

    // Pick a random subtree
    Expression &random_subtree = expression_picker.pick(result);

    // Replace the random subtree with a randomly generated expression
    random_subtree = expression_generator.generate();

    // Return the mutated expression
    result = expression_reorganizer.reorganize(result);

    // If the depth of the new expression exceeds the maximum depth, revert to original tree
    if (result.depth > config.max_depth) {
        result = expr;
    }

    // Return the mutated expression
    return result;
}

}