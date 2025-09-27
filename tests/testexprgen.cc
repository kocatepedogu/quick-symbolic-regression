// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "genetic/common/expression_generator.hpp"

#include <iostream>

using namespace qsr;

int main(void) {
    Config config(
    2, 3, 10, 11200, 0.01, 0.3, 0.1,
    std::make_shared<FunctionSet>(
        std::vector<std::string>{"+", "-", "*", "/", "sin", "cos", "exp", "relu"}),
        false
    );

    ExpressionGenerator expression_generator(config);

    for (int i = 0; i < 1000; i++) {
        std::cout << expression_generator.generate(10, false) << std::endl;
    }

    return 0;
}