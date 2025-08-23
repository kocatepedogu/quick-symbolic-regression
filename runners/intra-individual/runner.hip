// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "runner.hpp"

#include "./program/program.hpp"

#include "../../vm/vm_debug.hpp"
#include "../../vm/vm_control.hpp"

#include "../../util/rng.hpp"

// Uncomment to enable buffer overflow checks
// #define CHECK_BUFFER_OVERFLOW

#ifdef CHECK_BUFFER_OVERFLOW
    #define DEBUGARGS ,\
        stack_length ,\
        intermediate_length
#else
    #define DEBUGARGS
#endif

using namespace qsr;
using namespace qsr::intra_individual;

constexpr int reduction_threads_per_block = 1024;

Runner::Runner(const int nweights) 
    : GPUBaseRunner(nweights) {}

__global__ static void
update_weights(Ptr1D<float> weight_grads, float &weight, float learning_rate) 
{
    // Shared memory for block-level reduction
    __shared__ float sharedMem[reduction_threads_per_block];
    
    // Thread and block indices
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sharedMem[tid] = (globalIdx < weight_grads.dim1) ? weight_grads[globalIdx] : 0.0f;
    __syncthreads();
    
    // Parallel reduction within the block
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedMem[tid] += sharedMem[tid + stride];
        }
        __syncthreads();
    }
    
    // First thread of each block writes block sum to output
    if (tid == 0) {
        atomicAdd(&weight, -learning_rate * sharedMem[0]);
    }
}

__global__ static void 
train(Ptr1D<Instruction> bytecode, 
      Ptr2D<float> X_d, 
      Ptr1D<float> y_d, 
      Ptr2D<float> stack_d, 
      Ptr2D<float> intermediate_d,
      Ptr1D<float> weights_d,
      Ptr2D<float> weights_grad_d,
      Ptr1D<float> loss_d,
      int stack_length, 
      int intermediate_length)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // Mask excess threads
    if (tid >= y_d.dim1) {
        return;
    }

    // Reset weight gradients that will be computed by this thread
    for (int j = 0; j < weights_d.dim1; ++j) {
        weights_grad_d[j,tid] = 0;
    }

    int program_counter = 0;
    int stack_pointer = 0;
    int intermediate_pointer = 0;

    const StackState s(stack_d, intermediate_d, stack_pointer, intermediate_pointer);
    const ControlState c(tid, tid, bytecode.dim1, bytecode, program_counter);
    const DataState d(X_d, y_d);
    const WeightState w(weights_d, weights_grad_d);

    // Forward propagate and evaluate loss
    vm_debug_print(tid, "Forward propagation");
    vm_control<FORWARD, INTRA_INDIVIDUAL, Ptr1D<float>>(c, d, s, w DEBUGARGS);

    // Print an empty line in between forward propagation output and backpropagation output
    vm_debug_print(tid, "");

    // Save squared difference as the loss
    loss_d[tid] = powf(stack_d[0,tid], 2);

    // Backpropagate
    /// TODO: Do not run in the first epoch
    vm_debug_print(tid, "Backpropagation");
    vm_control<BACK, INTRA_INDIVIDUAL, Ptr1D<float>>(c, d, s, w DEBUGARGS);
}

void Runner::run(Ptr1D<Instruction> bytecode, int code_length, int stack_length, int intermediate_length, std::shared_ptr<const Dataset> dataset, int epochs, float learning_rate) {
    for (int epoch = 0; epoch < epochs + 1; ++epoch) {
        // Run forward propagation and backpropagation to compute
        // weight gradients for each weight and for each data point
        launch_kernel(train, bytecode, dataset->X_d.ptr, dataset->y_d.ptr,
                      stack_d.ptr, intermediate_d.ptr,
                      weights_d.ptr, weights_grad_d.ptr, loss_d.ptr, stack_length, intermediate_length);

        // For every weight, sum gradient contributions from all data points using reduction
        // Apply gradient descent rule
        if (epoch < epochs) {
            for (int i = 0; i < nweights; ++i) {
                launch_kernel(update_weights, weights_grad_d.ptr(i), weights_d.ptr[i], learning_rate);
            }
        }
    }

    synchronize();
}

void Runner::calculate_kernel_dims(int m) {
    const int threads_per_block = min(m, hipState.props.maxThreadsPerBlock);
    blockDim = dim3(threads_per_block);

    const int nblocks = (m + threads_per_block - 1) / threads_per_block;
    gridDim = dim3(nblocks);

    int reduction_blocks_per_grid = (m + reduction_threads_per_block - 1) / 
        reduction_threads_per_block;

    reduction_grid_dim = dim3(reduction_blocks_per_grid);
    reduction_block_dim = dim3(reduction_threads_per_block);
}

void Runner::initialize_weights_and_losses(Expression &expression) {
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

void Runner::save_weights_and_losses(Expression &expression) {
    // Compute total loss
    float total_loss = 0;
    #pragma omp simd reduction(+:total_loss)
    for (int i = 0; i < loss_d.ptr.dim1; ++i) {
        total_loss += loss_d.ptr[i];
    }

    // Save loss value to the original expression
    expression.loss = total_loss;

    // Save weights to the original expression
    expression.weights.resize(weights_d.ptr.dim1);
    for (int j = 0; j < weights_d.ptr.dim1; ++j) {
        expression.weights[j] = weights_d.ptr[j];
    }
}

void Runner::run(std::vector<Expression>& population, std::shared_ptr<const Dataset> dataset, int epochs, float learning_rate) {
    calculate_kernel_dims(dataset->m);

    // Convert symbolic expressions to bytecode program
    Program program_pop(population);

    // Sequential loop over programs
    for (int i = 0; i < population.size(); ++i)
    {
        // Get bytecode and number of instructions
        auto code = program_pop.bytecode.ptr(i);
        auto code_length = program_pop.num_of_instructions.ptr[i];

        // Get stack and intermediate requirements
        auto stack_length = program_pop.stack_req.ptr[i];
        auto intermediate_length = program_pop.intermediate_req.ptr[i];

        // Resize arrays
        loss_d.resize(dataset->m);
        weights_d.resize(nweights);
        weights_grad_d.resize(nweights, dataset->m);
        stack_d.resize(stack_length, dataset->m);
        intermediate_d.resize(intermediate_length, dataset->m);

        // If the expression has no weights yet, initialize them randomly
        initialize_weights_and_losses(population[i]);

        // Do the work
        run(code, code_length, stack_length, intermediate_length, dataset, epochs, learning_rate);

        // Save weights and losses to the original expression
        save_weights_and_losses(population[i]);
    }
}
