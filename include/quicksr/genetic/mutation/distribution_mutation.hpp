// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef MUTATION_DISTRIBUTION_HPP
#define MUTATION_DISTRIBUTION_HPP

#include "base.hpp"

#include <memory>

namespace qsr {

/**
 * @brief Mutation strategy that varies based on a probability distribution
 *
 * @details
 * This mutation strategy applies one of several mutation strategies with a specified probability.
 * The mutation strategy is chosen randomly according to the given probabilities at every mutation operation.
 *
 * @param mutations A vector of mutation strategies to be applied.
 * @param probabilities A vector of probabilities corresponding to each mutation strategy.
 */
class DistributionMutation : public BaseMutation {
public:
    constexpr DistributionMutation(std::vector<std::shared_ptr<BaseMutation>> mutations, std::vector<double> probabilities) :
        mutations(mutations), probabilities(probabilities) {
            if (mutations.size() != probabilities.size()) {
                throw std::invalid_argument("Mutations and probabilities must have the same size.");
            }
        }

    std::shared_ptr<BaseMutator> get_mutator(const Config &config) override;

private:
    std::vector<std::shared_ptr<BaseMutation>> mutations;

    std::vector<double> probabilities;
};

}

#endif