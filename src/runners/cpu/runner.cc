// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "runners/cpu/runner.hpp"

#include "vm/vm_control.hpp"
#include "vm/vm_types.hpp"

#include "util/precision.hpp"
#include "util/rng.hpp"

// Uncomment to enable buffer overflow checks
// #define CHECK_BUFFER_OVERFLOW

#ifdef CHECK_BUFFER_OVERFLOW
    #define DEBUGARGS ,\
        stack_req ,\
        intermediate_req
#else
    #define DEBUGARGS
#endif

using namespace qsr;
using namespace qsr::cpu;

using ProgramIndividual = intra_individual::ProgramIndividual;

Runner::Runner(int nweights, const bool use_cache) :
    BaseRunner(nweights), use_cache(use_cache) {
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
    loss_d.ptr[0] = 0;
}

void Runner::update_weights(real learning_rate) {
    // For every weight
    for (int weight_idx = 0; weight_idx < nweights; ++weight_idx) {
        // Reduce weight gradients from all datapoints to a single gradient
        real total_grad = 0;
        #pragma omp simd reduction(+:total_grad)
        for (int tid = 0; tid < weights_grad_d.ptr.dim2; ++tid) {
            total_grad += weights_grad_d.ptr[weight_idx,tid];
        }

        // Apply gradient descent
        weights_d.ptr[weight_idx] -= learning_rate * total_grad;
    }
}

void Runner::train(const ProgramIndividual &p, std::shared_ptr<const Dataset> dataset, int epochs, real learning_rate) {
    for (int epoch = 0; epoch < epochs + 1; ++epoch) {
        // Reset gradients and losses
        reset_gradients_and_losses();

        // Sequential loop over datapoints
        for (int tid = 0; tid < dataset->m; ++tid) {
            int program_counter = 0;
            int stack_pointer = 0;
            int intermediate_pointer = 0;

            const StackState s(stack_d.ptr, intermediate_d.ptr, stack_pointer, intermediate_pointer);
            const ControlState c(tid, tid, p.num_of_instructions, p.bytecode, program_counter);
            const DataState d(dataset->X_d.ptr, dataset->y_d.ptr);
            const WeightState w(weights_d.ptr, weights_grad_d.ptr);

            // Forward propagate and evaluate loss
            vm_debug_print(tid, "Forward propagation");
            vm_control<FORWARD, INTRA_INDIVIDUAL, Ptr1D<real>>(c, d, s, w DEBUGARGS);

            // Print an empty line in between forward propagation output and backpropagation output
            vm_debug_print(tid, "");

            // Save squared difference divided by m [ (l/m)^2 * m = l^2/m ] as the loss
            const float squared_loss = (float)stack_d.ptr[0,tid] * (float)stack_d.ptr[0,tid];
            loss_d.ptr[0] += squared_loss * (float)dataset->m;

            if (epochs > 0) {
                vm_debug_print(tid, "Backpropagation");
                vm_control<BACK, INTRA_INDIVIDUAL, Ptr1D<real>>(c, d, s, w DEBUGARGS);
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
            weights_d.ptr[j] = 2.0_r * (real)(thread_local_rng() % RAND_MAX) / (real)RAND_MAX - 1_r;
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
    expression.loss = loss_d.ptr[0];

    // Write final weights back to the original expression
    expression.weights = std::vector<double>(weights_d.ptr.ptr, weights_d.ptr.ptr + nweights);
}

void Runner::run(std::vector<Expression>& population, std::shared_ptr<const Dataset> dataset, int epochs, double learning_rate) {
    // Convert symbolic expressions to bytecode program
    intra_individual::Program program(population);

    // Loop over programs
    for (const auto &p : program) {
        if (use_cache && population_cache.load(population[p.index])) {
            continue;
        }

        // Resize VM memory
        resize_arrays(p.stack_req, p.intermediate_req, 
            dataset->m, dataset->m);

        // Initialize weights
        initialize_weights(population[p.index]);

        // Apply gradient descent
        train(p, dataset, epochs, learning_rate);

        // Save weights and losses
        save_weights_and_losses(population[p.index]);

        if (use_cache) {
            population_cache.save(population[p.index]);
        }
    }
}
