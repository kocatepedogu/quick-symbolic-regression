// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "runner.hpp"

#include <cmath>
#include <hip/hip_runtime.h>

#include "../../../vm/vm_debug.hpp"
#include "../../../vm/vm_types.hpp"
#include "../../../vm/vm_control.hpp"
#include "../../../util/rng.hpp"
#include "../../../util/hip.hpp"
#include "../../../util/arrays/array1d.hpp"
#include "../../../util/arrays/array2d.hpp"

#include "./program/program.hpp"

#include <hip/hip_runtime.h>
#include <memory>

// Uncomment to enable buffer overflow checks
// #define CHECK_BUFFER_OVERFLOW

namespace qsr::inter_individual {
    Runner::Runner(int nweights) :
        nweights(nweights) {}

    template <typename... Debug>
    __global__
    void vm(Ptr2D<Instruction> bytecode, 
            const int max_num_of_instructions,
            const int num_of_individuals,
            const int m, 
            Ptr2D<float> X_d, 
            Ptr1D<float> y_d, 
            Ptr2D<float> stack_d, 
            Ptr2D<float> intermediate_d,
            Ptr2D<float> weights_d,
            Ptr2D<float> weights_grad_d,
            const int nweights,
            const int nepochs,
            const float learning_rate,
            Ptr1D<float> loss_d,
            Debug... debug)
    {
        const int tid = blockDim.x * blockIdx.x + threadIdx.x;

        if (tid < num_of_individuals) {
            float total_loss = 0;

            for (int epoch = 0; epoch < nepochs + 1; ++epoch) {
                // Reset weight gradients
                for (int weight_idx = 0; weight_idx < nweights; ++weight_idx) {
                    weights_grad_d[weight_idx][tid] = 0;
                }

                total_loss = 0;

                // For all datapoints, compute and sum weight gradients
                for (int datapoint_idx = 0; datapoint_idx < m; ++datapoint_idx) {
                    int program_counter = 0;
                    int stack_pointer = 0;
                    int intermediate_pointer = 0;

                    const StackState s(stack_d, intermediate_d, stack_pointer, intermediate_pointer);
                    const ControlState c(tid, datapoint_idx, max_num_of_instructions, bytecode, program_counter);
                    const DataState d(X_d, y_d);
                    const WeightState w(weights_d, weights_grad_d);

                    // Forward propagate and evaluate loss
                    vm_debug_print(tid, "Forward propagation");
                    vm_control<FORWARD, INTER_INDIVIDUAL, Ptr2D<float>>(c, d, s, w, debug...);

                    // Print an empty line in between forward propagation output and backpropagation output
                    vm_debug_print(tid, "");

                    // Add squared difference to total loss
                    total_loss += powf(stack_d[0,tid], 2);

                    // Backpropagate
                    if (epoch < nepochs) {
                        vm_debug_print(tid, "Backpropagation");
                        vm_control<BACK, INTER_INDIVIDUAL, Ptr2D<float>>(c, d, s, w, debug...);
                    }
                }

                // Apply weight updates (gradient descent)
                if (epoch < nepochs) {
                    for (int weight_idx = 0; weight_idx < nweights; ++weight_idx) {
                        weights_d[weight_idx,tid] -= learning_rate * weights_grad_d[weight_idx][tid];
                    }
                }
            }

            loss_d[tid] = total_loss;
        }
    }

    void Runner::run(const Program &program, std::shared_ptr<const Dataset> dataset, int epochs, float learning_rate, 
                     Array1D<float> &loss_d, Ptr2D<float> &weights_d) {
        const int threadsPerBlock = min(program.num_of_individuals, config.props.maxThreadsPerBlock);
        const int blocks = (program.num_of_individuals + threadsPerBlock - 1) / threadsPerBlock;

        dim3 gridDim(blocks);
        dim3 blockDim(threadsPerBlock);

        Array2D<float> stack_d(program.stack_req, program.num_of_individuals);
        Array2D<float> intermediate_d(program.intermediate_req, program.num_of_individuals);
        Array2D<float> weights_grad_d(nweights, program.num_of_individuals);

        hipLaunchKernelGGL(
            vm, gridDim, blockDim, 0, config.stream,
            program.bytecode.ptr, 
            program.max_num_of_instructions, 
            program.num_of_individuals,
            dataset->m, dataset->X_d.ptr, dataset->y_d.ptr,
            stack_d.ptr, intermediate_d.ptr,
            weights_d, weights_grad_d.ptr, nweights,
            epochs, learning_rate, loss_d.ptr

            // Optional arguments for buffer overflow checking
            #ifdef CHECK_BUFFER_OVERFLOW
            , program.stack_req, program.intermediate_req
            #endif
        );

        HIP_CALL(hipStreamSynchronize(config.stream));
    }

    void Runner::run(std::vector<Expression>& population, std::shared_ptr<const Dataset> dataset, int epochs, float learning_rate) {
        // Convert symbolic expression to bytecode program
        Program program_pop(population);

        // Initialize weights
        Array2D<float> weights_d(nweights, population.size());
        for (int i = 0; i < nweights; ++i) {
            for (int j = 0; j < population.size(); ++j) {
                const auto &expression = population[j];

                // If the expression has no weights yet, initialize them randomly
                if (expression.weights.empty()) {
                    weights_d.ptr[i,j] = 2 * (thread_local_rng() % RAND_MAX) / (float)RAND_MAX - 1;
                } 
                // If the expression already has weights, use them
                else {
                    weights_d.ptr[i,j] = expression.weights[i];
                }   
            }
        }

        // Initialize loss array with NaN's
        Array1D<float> loss_d(population.size());
        for (int j = 0; j < population.size(); ++j) {
            loss_d.ptr[j] = std::numeric_limits<float>::quiet_NaN();
        }

        // Run virtual machine
        run(program_pop, dataset, epochs, learning_rate, loss_d, weights_d.ptr);

        // Write loss values to the original expressions
        for (int i = 0; i < population.size(); ++i) {
            population[i].loss = loss_d.ptr[i];
        }

        // Write weight values to the original expressions
        for (int i = 0; i < population.size(); ++i) {
            population[i].weights.resize(nweights);
            for (int j = 0; j < nweights; ++j) {
                population[i].weights[j] = weights_d.ptr[j,i];
            }
        }
    }
}