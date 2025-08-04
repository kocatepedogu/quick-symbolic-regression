// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef CROSSOVER_DEFAULT_HPP
#define CROSSOVER_DEFAULT_HPP

#include <tuple>

#include "../../expressions/expression.hpp"
#include "../expression_picker.hpp"
#include "base.hpp"

namespace qsr {

class DefaultCrossover : public BaseCrossover {
public:
    constexpr DefaultCrossover(float crossover_probability) :
        crossover_probability(crossover_probability), expression_picker() {}

    std::tuple<Expression, Expression> crossover(Expression e1, Expression e2) noexcept;

private:
    ExpressionPicker expression_picker;

    float crossover_probability;
};

}

#endif