// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef MUTATOR_DISTRIBUTION_HPP
#define MUTATOR_DISTRIBUTION_HPP

#include "../../../expressions/expression.hpp"

#include <memory>
#include <vector>
#include <cassert>

#include "base.hpp"

namespace qsr {

class DistributionMutator : public BaseMutator {
public:
    constexpr DistributionMutator(std::vector<std::shared_ptr<BaseMutator>> mutators, std::vector<float> probabilities) :
        mutators(mutators), probabilities(probabilities) {
            assert(mutators.size() == probabilities.size());
        }

    Expression mutate(const Expression &expr) noexcept override;

private:
    const std::vector<std::shared_ptr<BaseMutator>> mutators;

    const std::vector<float> probabilities;
};

}

#endif