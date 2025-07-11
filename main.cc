// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "testdata.hpp"
#include "expression.hpp"
#include "binary.hpp"
#include "unary.hpp"
#include "compiler.hpp"
#include "dataset.hpp"
#include "vm.hpp"

#include <iostream>
#include <cmath>

int main(void) {
    float **X, *y;

    // Generate ground truth data
    generate_test_data(X, y, [](int x) { 
        return 2.5382 * cos(x)*x + x*x - 0.5; 
    });

    // Generate ground truth dataset
    Dataset dataset(X, y, 100, 1);

    // Input feature
    Expression x = Var(0);

    // Symbolic expression
    Expression f = 2.5382 * Cos(x)*x + x*x - 0.5;

    // Convert symbolic expression to bytecode program
    Program p = compile(f);

    // Print bytecode instructions
    std::cout << p << std::endl;

    // Run program
    forward_propagate(p, dataset);

    // Free data
    delete_test_data(X, y);

    return 0;
}