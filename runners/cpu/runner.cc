// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "runner.hpp"

#include "../intra-individual/program/program.hpp"

#include "../../vm/vm_control.hpp"
#include "../../vm/vm_types.hpp"

#include "../../util/rng.hpp"

// Uncomment to enable buffer overflow checks
// #define CHECK_BUFFER_OVERFLOW

#ifdef CHECK_BUFFER_OVERFLOW
    #define DEBUGARGS ,\
        stack_req ,\
        intermediate_req
#else
    #define DEBUGARGS
#endif

namespace qsr::cpu {
    Runner::Runner(int nweights) :
        nweights(nweights) {
        weights_d.resize(nweights);
    }

    void Runner::reset_gradients_and_losses() {
        // For every weight
        for (int weight_idx = 0; weight_idx < nweights; ++weight_idx) {
            // Set weight gradient from each datapoint to zero
            #pragma omp simd
            for (int tid = 0; tid < weights_grad_d.ptr.dim2; ++tid) {
                weights_grad_d.ptr[weight_idx,tid] = 0;
            }
        }

        // Set the most recent loss to zero
        loss = 0;
    }

    void Runner::update_weights(float learning_rate) {
        // For every weight
        for (int weight_idx = 0; weight_idx < nweights; ++weight_idx) {
            // Reduce weight gradients from all datapoints to a single gradient
            float total_grad = 0;
            #pragma omp simd reduction(+:total_grad)
            for (int tid = 0; tid < weights_grad_d.ptr.dim2; ++tid) {
                total_grad += weights_grad_d.ptr[weight_idx,tid];
            }

            // Apply gradient descent
            weights_d.ptr[weight_idx] -= learning_rate * total_grad;
        }
    }

    void Runner::train(Instruction *bytecode, int num_of_instructions, std::shared_ptr<const Dataset> dataset, int epochs, float learning_rate,
                        int stack_req, int intermediate_req) {
        for (int epoch = 0; epoch < epochs + 1; ++epoch) {
            // Reset gradients and losses
            reset_gradients_and_losses();

            // Sequential loop over datapoints
            for (int tid = 0; tid < dataset->m; ++tid) {
                int program_counter = 0;
                int stack_pointer = 0;
                int intermediate_pointer = 0;

                const StackState s(stack_d.ptr, intermediate_d.ptr, stack_pointer, intermediate_pointer);
                const ControlState c(tid, tid, num_of_instructions, bytecode, program_counter);
                const DataState d(dataset->X_d.ptr, dataset->y_d.ptr);
                const WeightState w(weights_d.ptr, weights_grad_d.ptr);

                // Forward propagate and evaluate loss
                vm_debug_print(tid, "Forward propagation");
                vm_control<FORWARD, INTRA_INDIVIDUAL, Ptr1D<float>>(c, d, s, w DEBUGARGS);

                // Print an empty line in between forward propagation output and backpropagation output
                vm_debug_print(tid, "");

                // Save squared difference as the loss
                loss += powf(stack_d.ptr[0,tid], 2);

                if (epochs > 0) {
                    vm_debug_print(tid, "Backpropagation");
                    vm_control<BACK, INTRA_INDIVIDUAL, Ptr1D<float>>(c, d, s, w DEBUGARGS);
                }
            }

            // Compute total gradients with reduction and apply gradient descent
            if (epochs > 0) {
                update_weights(learning_rate);
            }
        }
    }

    void Runner::initialize_weights(Expression& expression) {
        // If the expression has no weights yet, initialize them randomly
        if (expression.weights.empty()) {
            for (int j = 0; j < nweights; ++j) {
                weights_d.ptr[j] = 2 * (thread_local_rng() % RAND_MAX) / (float)RAND_MAX - 1;
            }
        } 
        // If the expression already has weights, use them
        else {
            for (int j = 0; j < nweights; ++j) {
                weights_d.ptr[j] = expression.weights[j];
            }
        }
    }

    void Runner::save_weights_and_losses(Expression& expression) {
        // Write loss back to the original expression
        expression.loss = loss;

        // Write final weights back to the original expression
        expression.weights = std::vector<float>(weights_d.ptr.ptr, weights_d.ptr.ptr + nweights);
    }

    void Runner::run(std::vector<Expression>& population, std::shared_ptr<const Dataset> dataset, int epochs, float learning_rate) {
        // Convert symbolic expressions to bytecode program
        intra_individual::Program program(population);

        // Resize the array for storing gradients to dataset size
        weights_grad_d.resize(nweights, dataset->m);

        // Loop over programs
        for (int i = 0; i < program.num_of_individuals; ++i) {
            // Get the corresponding expression
            auto& expression = population[i];

            // Get bytecode and number of instructions
            auto instructions = program.bytecode.ptr[i];
            auto num_instructions = program.num_of_instructions.ptr[i];

            // Get stack and intermediate requirements
            auto stack_req = program.stack_req.ptr[i];
            auto intermediate_req = program.intermediate_req.ptr[i];

            // Resize stack and intermediate arrays
            stack_d.resize(stack_req, dataset->m);
            intermediate_d.resize(intermediate_req, dataset->m);

            // Initialize weights
            initialize_weights(expression);

            // Apply gradient descent
            train(instructions, num_instructions, dataset, epochs, learning_rate, 
                  stack_req, intermediate_req);

            // Save weights and losses
            save_weights_and_losses(expression);
        }
    }
}