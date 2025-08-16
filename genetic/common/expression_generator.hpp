// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef EXPRESSION_GENERATOR_HPP
#define EXPRESSION_GENERATOR_HPP

#include "../../expressions/expression.hpp"

#include "function_set.hpp"
#include "config.hpp"

#include <memory>
#include <random>

namespace qsr {

class ExpressionGenerator {
public:
    ExpressionGenerator();
    ExpressionGenerator(const Config &config);

    Expression generate() noexcept;
    Expression generate(int max_depth) noexcept;

private:
    int random_operation(int max_depth) noexcept;

    std::discrete_distribution<> depth_one_distribution;
    std::discrete_distribution<> depth_two_distribution;

    Config config;
};
}

#endif