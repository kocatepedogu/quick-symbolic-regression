// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef RECOMBINER_DEFAULT_HPP
#define RECOMBINER_DEFAULT_HPP

#include <tuple>

#include "../../../expressions/expression.hpp"

#include "../../expression_picker.hpp"
#include "../../expression_reorganizer.hpp"

#include "base.hpp"

namespace qsr {

class DefaultRecombiner : public BaseRecombiner {
public:
    constexpr DefaultRecombiner(int max_depth, float crossover_probability) :
        max_depth(max_depth),
        crossover_probability(crossover_probability), 
        expression_reorganizer(),
        expression_picker() {}

    std::tuple<Expression, Expression> recombine(Expression e1, Expression e2) noexcept;

private:
    ExpressionPicker expression_picker;

    ExpressionReorganizer expression_reorganizer;

    const int max_depth;

    float crossover_probability;
};

}

#endif