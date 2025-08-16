// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef EXPRESSION_GENERATOR_HPP
#define EXPRESSION_GENERATOR_HPP

#include "../../expressions/expression.hpp"

namespace qsr {

class ExpressionGenerator {
public:
    constexpr ExpressionGenerator(int nvars, int nweights, int max_depth) : 
        nvars(nvars), nweights(nweights), max_depth(max_depth) {}

    Expression generate() const noexcept;

private:
    const int nvars;
    const int nweights;
    const int max_depth;

    int random_operation(int max_depth) const noexcept;

    Expression generate(int max_depth) const noexcept;
};

}

#endif