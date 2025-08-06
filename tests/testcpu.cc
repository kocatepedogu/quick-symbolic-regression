// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/testdata.hpp"

#include "../expressions/expression.hpp"
#include "../expressions/binary.hpp"
#include "../expressions/unary.hpp"

#include "../dataset/dataset.hpp"
#include "../runners/cpu/runner.hpp"

#include <iostream>

using namespace qsr;

int main(void) {
    float **X, *y;

    // Generate ground truth data
    generate_test_data(X, y, [](float x) { 
        return 2.5382 * cos(x)*x + x*x - 0.5; 
    });

    // Create dataset
    auto dataset = std::make_shared<Dataset>(X, y, test_data_length, 1);

    // Input feature
    Expression x = Var(0);

    // Trainable parameters
    Expression w0 = Parameter(0);
    Expression w1 = Parameter(1);
    Expression w2 = Parameter(2);

    // Symbolic expression
    Expression f1 = w0 * Cos(x)*x + x*x - w1;
    Expression f2 = w0 * Cos(x) + x - w1;
    Expression f3 = w0 * Sin(x + w2)*x + x*x - w1;

    // Construct a population
    std::vector<Expression> expression_pop = {f1, f2, f3};

    // Fit expressions
    cpu::Runner runner(dataset, 3);

    for (int i = 0; i < 60; ++i) {
        runner.run(expression_pop, 10, 1e-3);
    }

    // Print losses
    std::cout << "CPU" << std::endl;
    std::cout << "f1: " << expression_pop[0].loss << std::endl;
    std::cout << "f2: " << expression_pop[1].loss << std::endl;
    std::cout << "f3: " << expression_pop[2].loss << std::endl;

    // Free data
    delete_test_data(X, y);

    return 0;
}