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
        weights_grad_d.resize(nweights, initial_array_size);
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

    void Runner::reset_gradients() {
        // For every weight
        for (int weight_idx = 0; weight_idx < nweights; ++weight_idx) {
            // Set weight gradient from each datapoint to zero
            #pragma omp simd
            for (int tid = 0; tid < weights_grad_d.ptr.dim2; ++tid) {
                weights_grad_d.ptr[weight_idx,tid] = 0;
            }
        }
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

    float Runner::train(Instruction *bytecode, int num_of_instructions, std::shared_ptr<const Dataset> dataset, int epochs, float learning_rate,
                        int stack_req, int intermediate_req) {
        float final_loss = 0;

        for (int epoch = 0; epoch < epochs + 1; ++epoch) {
            float loss = 0;

            // Reset gradients
            reset_gradients();

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

            // Save current loss
            final_loss = loss;

            // Compute total gradients with reduction and apply gradient descent
            if (epochs > 0) {
                update_weights(learning_rate);
            }
        }

        return final_loss;
    }

    void Runner::run(std::vector<Expression>& population, std::shared_ptr<const Dataset> dataset, int epochs, float learning_rate) {
        // Convert symbolic expressions to bytecode program
        intra_individual::Program program_pop(population);

        // Resize the array for storing gradients to dataset size
        weights_grad_d.resize(nweights, dataset->m);

        // Loop over programs
        for (int program_idx = 0; program_idx < program_pop.num_of_individuals; ++program_idx) {
            // Resize stack and intermediate arrays for the current program
            stack_d.resize(program_pop.stack_req.ptr[program_idx], dataset->m);
            intermediate_d.resize(program_pop.intermediate_req.ptr[program_idx], dataset->m);

            // Initialize weights for the current program
            initialize_weights(population[program_idx]);

            // Get bytecode and number of instructions for the current program
            auto instructions = program_pop.bytecode.ptr[program_idx];
            auto num_instructions = program_pop.num_of_instructions.ptr[program_idx];

            // Get stack and intermediate requirements for the current program
            auto stack_req = program_pop.stack_req.ptr[program_idx];
            auto intermediate_req = program_pop.intermediate_req.ptr[program_idx];

            // Apply gradient descent for the current program
            float loss = train(instructions, num_instructions, dataset, epochs, learning_rate, 
                               stack_req, intermediate_req);

            // Write loss back to the original expression
            population[program_idx].loss = loss;

            // Write final weights back to the original expression
            population[program_idx].weights = std::vector<float>(weights_d.ptr.ptr, weights_d.ptr.ptr + nweights);
        }
    }
}