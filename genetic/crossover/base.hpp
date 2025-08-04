// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef CROSSOVER_BASE_HPP_DEFINED
#define CROSSOVER_BASE_HPP_DEFINED

#include "../../expressions/expression.hpp"

#include <tuple>

namespace qsr {

class BaseCrossover {
public:
    virtual std::tuple<Expression, Expression> crossover(Expression e1, Expression e2) noexcept;

    virtual ~BaseCrossover() = default;
};

}

#endif