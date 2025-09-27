// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef MUTATION_HOIST_HPP
#define MUTATION_HOIST_HPP

#include "base.hpp"

#include <memory>

namespace qsr {

/**
 * @brief Hoist mutation strategy
 *
 * @details
 * This mutation strategy randomly selects a subtree from the expression and replaces it
 * with a randomly chosen, smaller subtree within that subtree. This can lead to expressions
 * becoming more simple, fighting the bloat problem.
 *
 * @param mutation_probability The probability of performing the mutation on an individual.
 */
class HoistMutation : public BaseMutation {
public:
    constexpr HoistMutation(double mutation_probability) :
        mutation_probability(mutation_probability) {}

    virtual std::shared_ptr<BaseMutator> get_mutator(const Config &config) override;

private:
    double mutation_probability;
};

}

#endif