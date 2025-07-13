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
    generate_test_data(X, y, [](float x) { 
        return 2.5382 * cos(x)*x + x*x - 0.5; 
    });

    // Generate ground truth dataset
    Dataset dataset(X, y, 1000, 1);

    // Input feature
    Expression x = Var(0);

    // Trainable parameters
    Expression w0 = Parameter(0);
    Expression w1 = Parameter(1);

    // Symbolic expression
    Expression f = w0 * Cos(x)*x + x*x - w1;

    // Convert symbolic expression to bytecode program
    Program p = compile(f);

    // Print bytecode instructions
    std::cout << p << std::endl;

    // Create virtual machine
    VirtualMachine vm(dataset, 2);

    // Run program
    vm.fit(p);

    // Free data
    delete_test_data(X, y);

    return 0;
}