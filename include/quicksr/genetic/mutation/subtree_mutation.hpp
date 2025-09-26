// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef MUTATION_SUBTREE_HPP
#define MUTATION_SUBTREE_HPP

#include "base.hpp"

#include <memory>

namespace qsr {

/**
 * @brief Subtree mutation strategy
 *
 * @details
 * This mutation strategy randomly selects a subtree in the expression.
 * The selected subtree is then replaced by another randomly generated subtree.
 * This allows for significant changes in the expression structure, potentially leading to new solutions.
 *
 * The `max_depth_increment` parameter controls the maximum increase in depth allowed for the new subtree.
 * This helps in controlling the complexity of the expressions and prevents excessive growth.
 *
 * The `mutation_probability` parameter determines the likelihood of applying the mutation to an individual.
 *
 * @param max_depth_increment The maximum increase in depth allowed for the new subtree.
 * @param mutation_probability The probability of applying the mutation to an individual.
 */
class SubtreeMutation : public BaseMutation {
public:
    constexpr SubtreeMutation(int max_depth_increment, double mutation_probability) :
        max_depth_increment(max_depth_increment),
        mutation_probability(mutation_probability) {}

    virtual std::shared_ptr<BaseMutator> get_mutator(const Config &config) override;

private:
    int max_depth_increment;

    double mutation_probability;
};

}

#endif