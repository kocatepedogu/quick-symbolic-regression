// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "./tests/testdata.hpp"

#include "./expressions/expression.hpp"
#include "./expressions/binary.hpp"
#include "./expressions/unary.hpp"

#include "./intra-individual/dataset/dataset.hpp"
#include "./intra-individual/compiler/programpopulation.hpp"
#include "./intra-individual/vm/vm.hpp"

#include "util.hpp"

#include <cmath>
#include </usr/lib/clang/20/include/omp.h>

omp_lock_t lock;
omp_lock_t print_lock;

int main(void) {
    omp_init_lock(&lock);
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

    // Construct a population
    std::vector<Expression> expression_pop;
    for (int i = 0; i < 100; ++i) {
        expression_pop.push_back(f);
    }

    // Convert symbolic expression to bytecode program
    ProgramPopulation program_pop = compile(expression_pop);

    // Number of streams
    constexpr int nstreams = 2;

    // Remaining work
    int work_count = expression_pop.size();

    // Parallel region
    #pragma omp parallel num_threads(nstreams)
    {
        // Stream of the thread
        hipStream_t stream;
        HIP_CALL(hipStreamCreate(&stream));

        // Virtual machine of the thread
        VirtualMachine *vm = new VirtualMachine(dataset, stream, 2, print_lock);

        // Fit
        while(true) {
            omp_set_lock(&lock);
            if (work_count == 0) {
                omp_unset_lock(&lock);
                break;
            } else {
                const Program *p = program_pop.individuals[work_count - 1];
                --work_count;
                omp_unset_lock(&lock);
                vm->fit(*p);
            }
        }

        // Delete machine
        delete vm;

        // Delete stream
        HIP_CALL(hipStreamDestroy(stream));
    }

    // Free data
    delete_test_data(X, y);

    // Remove locks
    omp_destroy_lock(&lock);
    omp_destroy_lock(&print_lock);

    return 0;
}