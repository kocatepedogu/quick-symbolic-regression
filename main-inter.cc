// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "./tests/testdata.hpp"

#include "./expressions/expression.hpp"
#include "./expressions/binary.hpp"
#include "./expressions/unary.hpp"

#include "./dataset/dataset.hpp"
#include "inter-individual/runner.hpp"

#include <cmath>

int main(void) {
    float **X, *y;

    // Generate ground truth data
    generate_test_data(X, y, [](float x) { 
        return 2.5382 * cos(x)*x + x*x - 0.5; 
    });

    // Create dataset
    Dataset dataset(X, y, test_data_length, 1);

    // Input feature
    Expression x = Var(0);

    // Trainable parameters
    Expression w0 = Parameter(0);
    Expression w1 = Parameter(1);

    // Symbolic expression
    Expression f = w0 * Cos(x)*x + x*x - w1;

    // Construct a population
    std::vector<Expression> expression_pop;
    for (int i = 0; i < 11200; ++i) {
        expression_pop.push_back(f);
    }

    // Fit expressions
    inter_individual::Runner runner;
    runner.run(expression_pop, dataset);

    // Free data
    delete_test_data(X, y);

    return 0;
}