// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "./tests/testdata.hpp"

#include "./expressions/expression.hpp"
#include "./expressions/binary.hpp"
#include "./expressions/unary.hpp"

#include "./dataset/dataset.hpp"
#include "inter-individual/runner.hpp"

#include <cmath>
#include <iostream>

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
    Expression w2 = Parameter(2);

    // Symbolic expression
    Expression f1 = w0 * Cos(x)*x + x*x - w1;
    Expression f2 = w0 * Cos(x) + x - w1;
    Expression f3 = w0 * Sin(x + w2)*x + x*x - w1;

    // Construct a population
    std::vector<Expression> expression_pop = {f1, f2, f3,};

    // Fit expressions
    inter_individual::Runner runner(dataset, 3);
    runner.run(expression_pop);

    // Print fitnesses
    std::cout << "f1: " << expression_pop[0].fitness << std::endl;
    std::cout << "f2: " << expression_pop[1].fitness << std::endl;
    std::cout << "f3: " << expression_pop[2].fitness << std::endl;

    // Free data
    delete_test_data(X, y);

    return 0;
}