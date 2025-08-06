// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "runner.hpp"

#include "../intra-individual/program/program.hpp"

#include "../../vm/vm_control.hpp"
#include "../../vm/vm_types.hpp"

#include "../../util/hip.hpp"
#include "../../util/rng.hpp"

namespace qsr {
extern int free_core_counter;

namespace cpu {
    constexpr int max_stack_depth = 128;

    Runner::Runner(std::shared_ptr<Dataset> dataset, int nweights) :
        dataset(dataset), nweights(nweights) {}

    void Runner::run(std::vector<Expression>& population, int epochs, float learning_rate) {
        // Convert symbolic expressions to bytecode program
        intra_individual::Program program_pop(population);

        Array2D<float> stack_d(max_stack_depth, dataset->m);
        Array2D<float> intermediate_d(max_stack_depth, dataset->m);
        Array1D<float> weights_d(nweights);
        Array2D<float> weights_grad_d(nweights, dataset->m);

        // Loop over programs
        #pragma omp parallel
        {
            // Iterate over programs
            #pragma omp for
            for (int program_idx = 0; program_idx < program_pop.num_of_individuals; ++program_idx) {
                // If the expression has no weights yet, initialize them randomly
                if (population[program_idx].weights.empty()) {
                    for (int j = 0; j < nweights; ++j) {
                        weights_d.ptr[j] = 2 * (thread_local_rng() % RAND_MAX) / (float)RAND_MAX - 1;
                    }
                } 
                // If the expression already has weights, use them
                else {
                    for (int j = 0; j < nweights; ++j) {
                        weights_d.ptr[j] = population[program_idx].weights[j];
                    }
                }

                for (int epoch = 0; epoch < epochs; ++epoch) {
                    float loss = 0;

                    // Reset gradients
                    for (int weight_idx = 0; weight_idx < nweights; ++weight_idx) {
                        #pragma omp simd
                        for (int tid = 0; tid < dataset->m; ++tid) {
                            weights_grad_d.ptr[weight_idx,tid] = 0;
                        }
                    }

                    // Sequential loop over datapoints (parallelize with AVX if possible)
                    #pragma omp simd reduction(+:loss)
                    for (int tid = 0; tid < dataset->m; ++tid) {
                        int stack_pointer = 0;
                        int intermediate_pointer = 0;

                        const StackState s(
                            stack_d.ptr,
                            intermediate_d.ptr,
                            stack_pointer,
                            intermediate_pointer
                        );

                        int program_counter = 0;

                        // Forward propagate and evaluate loss
                        vm_debug_print(tid, "Forward propagation");
                        vm_control<FORWARD, INTRA_INDIVIDUAL, c_inst_1d, Ptr1D<float>>(
                            tid, tid, program_pop.bytecode.ptr[program_idx], program_pop.num_of_instructions.ptr[program_idx], 
                            dataset->m, dataset->X_d.ptr, dataset->y_d.ptr, 
                            s, program_counter, weights_d.ptr, weights_grad_d.ptr);

                        // Print an empty line in between forward propagation output and backpropagation output
                        vm_debug_print(tid, "");

                        // Save squared difference as the loss
                        loss += powf(stack_d.ptr[0,tid], 2);

                        vm_debug_print(tid, "Backpropagation");
                        vm_control<BACK, INTRA_INDIVIDUAL, c_inst_1d, Ptr1D<float>>(
                            tid, tid, program_pop.bytecode.ptr[program_idx], program_pop.num_of_instructions.ptr[program_idx], 
                            dataset->m, dataset->X_d.ptr, dataset->y_d.ptr, 
                            s, program_counter, weights_d.ptr, weights_grad_d.ptr);
                    }

                    // Write loss to the original expression
                    population[program_idx].loss = loss;

                    // Compute total gradients with reduction and apply gradient descent
                    for (int weight_idx = 0; weight_idx < nweights; ++weight_idx) {
                        // Reduce gradients
                        float total_grad = 0;
                        #pragma omp simd reduction(+:total_grad)
                        for (int tid = 0; tid < dataset->m; ++tid) {
                            total_grad += weights_grad_d.ptr[weight_idx,tid];
                        }

                        // Apply gradient descent
                        weights_d.ptr[weight_idx] -= learning_rate * total_grad;
                    }
                }

                // Write final weights back to the original expression
                population[program_idx].weights = std::vector<float>(weights_d.ptr.ptr, weights_d.ptr.ptr + nweights);
            }
        }

    }
}}