// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef EXPRESSION_GENERATOR_FULL_HPP
#define EXPRESSION_GENERATOR_FULL_HPP

#include "../../expressions/expression.hpp"
#include "function_set.hpp"

#include <memory>
#include <random>

namespace qsr {

class FullExpressionGenerator {
public:
    FullExpressionGenerator(int nvars, int nweights, int depth, std::shared_ptr<FunctionSet> function_set);

    Expression generate(int remaining_depth) noexcept;

    Expression generate() noexcept;

private:
    const int nvars;
    const int nweights;
    const int depth;

    std::shared_ptr<FunctionSet> function_set;
    std::discrete_distribution<> depth_one_distribution;
    std::discrete_distribution<> depth_two_distribution;
};

}

#endif