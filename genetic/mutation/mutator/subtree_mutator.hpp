// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef MUTATOR_DEFAULT_HPP
#define MUTATOR_DEFAULT_HPP

#include "../../../expressions/expression.hpp"

#include "../../expression_generator.hpp"
#include "../../expression_picker.hpp"
#include "../../expression_reorganizer.hpp"

#include "base.hpp"

namespace qsr {

class SubtreeMutator : public BaseMutator {
public:
    constexpr SubtreeMutator(int nvars, int nweights, 
                            int max_depth_increment, int max_depth, 
                            float mutation_probability)  :
        max_depth(max_depth),
        mutation_probability(mutation_probability),
        expression_generator(nvars, nweights, max_depth_increment) {}

    Expression mutate(const Expression &expr) noexcept;

private:
    const int max_depth;

    const float mutation_probability;

    ExpressionGenerator expression_generator;

    ExpressionPicker expression_picker;

    ExpressionReorganizer expression_reorganizer;
};

}

#endif