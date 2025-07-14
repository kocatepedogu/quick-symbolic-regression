// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "testdata.hpp"
#include "expression.hpp"
#include "binary.hpp"
#include "unary.hpp"
#include "compiler.hpp"
#include "dataset.hpp"
#include "util.hpp"
#include "vm.hpp"

#include <iostream>
#include <cmath>

#include </usr/lib/clang/20/include/omp.h>

omp_lock_t print_lock;

int main(void) {
    omp_init_lock(&print_lock);

    float **X, *y;

    // Generate ground truth data
    generate_test_data(X, y, [](float x) { 
        return 2.5382 * cos(x)*x + x*x - 0.5; 
    });

    // Generate ground truth dataset
    Dataset dataset(X, y, test_data_length, 1);

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

    // Number of streams
    constexpr int nstreams = 2;

    // Parallel region
    #pragma omp parallel num_threads(nstreams)
    {
        // Stream of the thread
        hipStream_t stream;
        HIP_CALL(hipStreamCreate(&stream));

        // Virtual machine of the thread
        VirtualMachine *vm = new VirtualMachine(dataset, stream, 2, print_lock);

        // Fit
        for (int j = 0; j < 1000 / nstreams; ++j) {
            vm->fit(p);
        }

        // Delete machine
        delete vm;

        // Delete stream
        HIP_CALL(hipStreamDestroy(stream));
    }

    // Free data
    delete_test_data(X, y);

    return 0;
}