// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef EXPRESSION_GENERATOR_HPP
#define EXPRESSION_GENERATOR_HPP

#include "../../expressions/expression.hpp"
#include "function_set.hpp"

#include <memory>
#include <random>

namespace qsr {

class ExpressionGenerator {
public:
    ExpressionGenerator(int nvars, int nweights, int max_depth, std::shared_ptr<FunctionSet> function_set);

    Expression generate() noexcept;
    Expression generate(int max_depth) noexcept;

private:
    const int nvars;
    const int nweights;
    const int max_depth;

    int random_operation(int max_depth) noexcept;

    std::shared_ptr<FunctionSet> function_set;
    std::discrete_distribution<> depth_one_distribution;
    std::discrete_distribution<> depth_two_distribution;
};

}

#endif