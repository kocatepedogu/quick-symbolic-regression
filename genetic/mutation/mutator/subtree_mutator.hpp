// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef MUTATOR_DEFAULT_HPP
#define MUTATOR_DEFAULT_HPP

#include "../../../expressions/expression.hpp"
#include "../../recombination/recombiner/default.hpp"
#include "../../expression_generator.hpp"
#include "base.hpp"

namespace qsr {

class SubtreeMutator : public BaseMutator {
public:
    constexpr SubtreeMutator(int nvars, int nweights, 
                            int max_depth_increment, int max_depth, 
                            float mutation_probability)  :
        max_depth(max_depth),
        expression_generator(nvars, nweights, max_depth_increment), 
        recombiner(max_depth, mutation_probability) {}

    Expression mutate(const Expression &expr) noexcept;

private:
    const int max_depth;

    ExpressionGenerator expression_generator;

    DefaultRecombiner recombiner;
};

}

#endif