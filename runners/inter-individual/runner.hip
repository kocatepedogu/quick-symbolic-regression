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

#ifdef CHECK_BUFFER_OVERFLOW
    #define DEBUGARGS ,\
        program.stack_req ,\
        program.intermediate_req
#else
    #define DEBUGARGS
#endif

using namespace qsr;
using namespace qsr::inter_individual;

Runner::Runner(int nweights) :
    GPUBaseRunner(nweights) {}

static inline __device__ void
reset_gradients_and_losses(const int tid, Ptr2D<float> weights_grad_d, float& total_loss) 
{
    // Reset weight gradients
    for (int weight_idx = 0; weight_idx < weights_grad_d.dim1; ++weight_idx) {
        weights_grad_d[weight_idx][tid] = 0;
    }

    total_loss = 0;
}

static inline __device__ void
update_weights(const int tid, Ptr2D<float> weights_d, Ptr2D<float> weights_grad_d, const float learning_rate) 
{
    // For each weight
    for (int weight_idx = 0; weight_idx < weights_d.dim1; ++weight_idx) {
        // Apply gradient descent
        weights_d[weight_idx,tid] -= learning_rate * weights_grad_d[weight_idx][tid];
    }
}

template <typename... Debug>
__global__ static void 
train(Ptr2D<Instruction> bytecode,
      Ptr2D<float> X_d, Ptr1D<float> y_d, 
      Ptr2D<float> stack_d, Ptr2D<float> intermediate_d,
      Ptr2D<float> weights_d, Ptr2D<float> weights_grad_d, Ptr1D<float> loss_d,
      const int nepochs, const float learning_rate, Debug... debug)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < bytecode.dim2) {
        float total_loss = 0;

        for (int epoch = 0; epoch < nepochs + 1; ++epoch) {
            // At the beginning of each epoch, reset gradients and losses
            reset_gradients_and_losses(tid, weights_grad_d, total_loss);

            // For all datapoints, compute and sum weight gradients
            for (int datapoint_idx = 0; datapoint_idx < y_d.dim1; ++datapoint_idx) {
                int program_counter = 0;
                int stack_pointer = 0;
                int intermediate_pointer = 0;

                const StackState s(stack_d, intermediate_d, stack_pointer, intermediate_pointer);
                const ControlState c(tid, datapoint_idx, bytecode.dim1, bytecode, program_counter);
                const DataState d(X_d, y_d);
                const WeightState w(weights_d, weights_grad_d);

                // Forward propagate and evaluate loss
                vm_debug_print(tid, "Forward propagation");
                vm_control<FORWARD, INTER_INDIVIDUAL, Ptr2D<float>>(c, d, s, w);

                // Print an empty line in between forward propagation output and backpropagation output
                vm_debug_print(tid, "");

                // Add squared difference to total loss
                total_loss += powf(stack_d[0,tid], 2);

                // Backpropagate
                if (epoch < nepochs) {
                    vm_debug_print(tid, "Backpropagation");
                    vm_control<BACK, INTER_INDIVIDUAL, Ptr2D<float>>(c, d, s, w);
                }
            }

            // At the end of each epoch, apply weight updates (gradient descent)
            if (epoch < nepochs) {
                update_weights(tid, weights_d, weights_grad_d, learning_rate);
            }
        }

        loss_d[tid] = total_loss;
    }
}

void Runner::calculate_kernel_dims(const Program &program) {
    const int threadsPerBlock = min(program.num_of_individuals, hipState.props.maxThreadsPerBlock);
    const int blocks = (program.num_of_individuals + threadsPerBlock - 1) / threadsPerBlock;

    gridDim = dim3(blocks);
    blockDim = dim3(threadsPerBlock);
}

void Runner::initialize_weights_and_losses(std::vector<Expression>& population) {
    // Initialize weights
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
    for (int j = 0; j < population.size(); ++j) {
        loss_d.ptr[j] = std::numeric_limits<float>::quiet_NaN();
    }
}

void Runner::save_weights_and_losses(std::vector<Expression>& population) {
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

void Runner::run(std::vector<Expression>& population, std::shared_ptr<const Dataset> dataset, int epochs, float learning_rate) {
    // Convert symbolic expression to bytecode program
    Program program(population);

    // Calculate kernel launch parameters
    calculate_kernel_dims(program);

    // Resize arrays
    loss_d.resize(population.size());
    weights_d.resize(nweights, population.size());
    weights_grad_d.resize(nweights, population.size());
    stack_d.resize(program.stack_req, population.size());
    intermediate_d.resize(program.intermediate_req, population.size());

    // Initialize weights and losses
    initialize_weights_and_losses(population);
    
    // Run virtual machine
    hipLaunchKernelGGL(
        train, gridDim, blockDim, 0, hipState.stream,
        program.bytecode.ptr, 
        dataset->X_d.ptr, dataset->y_d.ptr,
        stack_d.ptr, intermediate_d.ptr,
        weights_d.ptr, weights_grad_d.ptr, loss_d.ptr,
        epochs, learning_rate DEBUGARGS
    );

    // Wait for the kernel to finish
    HIP_CALL(hipStreamSynchronize(hipState.stream));

    // Save weights and losses
    save_weights_and_losses(population);
}