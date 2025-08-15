// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef MUTATION_DISTRIBUTION_HPP
#define MUTATION_DISTRIBUTION_HPP

#include "base.hpp"

#include <memory>

namespace qsr {

class DistributionMutation : public BaseMutation {
public:
    constexpr DistributionMutation(std::vector<std::shared_ptr<BaseMutation>> mutations, std::vector<float> probabilities) :
        mutations(mutations), probabilities(probabilities) {
            if (mutations.size() != probabilities.size()) {
                throw std::invalid_argument("Mutations and probabilities must have the same size.");
            }
        }

    std::shared_ptr<BaseMutator> get_mutator(int nvars, int nweights, int max_depth) override;

private:
    std::vector<std::shared_ptr<BaseMutation>> mutations;

    std::vector<float> probabilities;
};

}

#endif