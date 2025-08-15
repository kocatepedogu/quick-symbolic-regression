// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "runner.hpp"

#include "./program/program.hpp"

#include "../../vm/vm_debug.hpp"
#include "../../vm/vm_control.hpp"

#include "../../util/rng.hpp"

// Uncomment to enable buffer overflow checks
// #define CHECK_BUFFER_OVERFLOW

namespace qsr::intra_individual {
    constexpr int reduction_threads_per_block = 1024;

    __global__ 
    void weight_update(c_real_1d weight_grads, real_1d weight, int m, float learning_rate) {
        // Shared memory for block-level reduction
        __shared__ float sharedMem[reduction_threads_per_block];
        
        // Thread and block indices
        int tid = threadIdx.x;
        int bid = blockIdx.x;
        int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
        
        // Load data into shared memory
        sharedMem[tid] = (globalIdx < m) ? weight_grads[globalIdx] : 0.0f;
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
            atomicAdd(weight, -learning_rate * sharedMem[0]);
        }
    }

    template <typename... Debug>
    __global__
    void vm(c_inst_1d bytecode, 
            const int bytecode_length,
            const int m, 
            Ptr2D<float> X_d, 
            Ptr1D<float> y_d, 
            Ptr2D<float> stack_d, 
            Ptr2D<float> intermediate_d,
            Ptr1D<float> weights_d,
            Ptr2D<float> weights_grad_d,
            const int nweights,
            Ptr1D<float> loss_d,
            Debug ...debug)
    {
        const int tid = blockDim.x * blockIdx.x + threadIdx.x;

        // Mask excess threads
        if (tid >= m) {
            return;
        }

        // Reset weight gradients that will be computed by this thread
        for (int j = 0; j < nweights; ++j) {
            weights_grad_d[j,tid] = 0;
        }

        int stack_pointer = 0;
        int intermediate_pointer = 0;

        const StackState s(
            stack_d,
            intermediate_d,
            stack_pointer,
            intermediate_pointer
        );

        int program_counter = 0;

        // Forward propagate and evaluate loss
        vm_debug_print(tid, "Forward propagation");
        vm_control<FORWARD, INTRA_INDIVIDUAL, c_inst_1d, Ptr1D<float>>(
            tid, tid, bytecode, bytecode_length, 
            m, X_d, y_d, 
            s, program_counter, weights_d, weights_grad_d, debug...);

        // Print an empty line in between forward propagation output and backpropagation output
        vm_debug_print(tid, "");

        // Save squared difference as the loss
        loss_d[tid] = powf(stack_d[0,tid], 2);

        // Backpropagate
        vm_debug_print(tid, "Backpropagation");
        vm_control<BACK, INTRA_INDIVIDUAL, c_inst_1d, Ptr1D<float>>(
            tid, tid, bytecode, bytecode_length, 
            m, X_d, y_d, 
            s, program_counter, 
            weights_d, weights_grad_d, debug...);
    }

    Runner::Runner(std::shared_ptr<const Dataset> dataset, const int nweights) : dataset(dataset), nweights(nweights) {
        const int threadsPerBlock = min(dataset->m, hipState.props.maxThreadsPerBlock);
        const int blocks = (dataset->m + threadsPerBlock - 1) / threadsPerBlock;

        gridDim = dim3(blocks);
        blockDim = dim3(threadsPerBlock);

        int reduction_blocks_per_grid = (dataset->m + reduction_threads_per_block - 1) / 
            reduction_threads_per_block;

        reduction_grid_dim = dim3(reduction_blocks_per_grid);
        reduction_block_dim = dim3(reduction_threads_per_block);

        weights_d = Array1D<float>(nweights);
        weights_grad_d = Array2D<float>(nweights, dataset->m);
        loss_d = Array1D<float>(dataset->m);
    }

    void Runner::run(c_inst_1d code, int code_length, int stack_length, int intermediate_length, int epochs, float learning_rate) {
        for (int epoch = 0; epoch < epochs + 1; ++epoch) {
            // Run forward propagation and backpropagation to compute
            // weight gradients for each weight and for each data point
            hipLaunchKernelGGL(
                vm, gridDim, blockDim, 0, hipState.stream,
                code, code_length, 
                dataset->m, dataset->X_d.ptr, dataset->y_d.ptr,
                stack_d.ptr, intermediate_d.ptr,
                weights_d.ptr, weights_grad_d.ptr, nweights, loss_d.ptr

                // Optional arguments for buffer overflow checking
                #ifdef CHECK_BUFFER_OVERFLOW
                , stack_length, intermediate_length
                #endif
            );

            // For every weight, sum gradient contributions from all data points using reduction
            // Apply gradient descent rule
            if (epoch < epochs) {
                for (int i = 0; i < nweights; ++i) {
                    hipLaunchKernelGGL(
                        weight_update, reduction_grid_dim, reduction_block_dim, 0, hipState.stream, 
                        weights_grad_d.ptr[i], &weights_d.ptr[i], dataset->m, learning_rate);
                }
            }
        }

        HIP_CALL(hipStreamSynchronize(hipState.stream));
    }

    void Runner::run(std::vector<Expression>& population, int epochs, float learning_rate) {
        // Convert symbolic expressions to bytecode program
        Program program_pop(population);

        // Sequential loop over programs
        for (int i = 0; i < population.size(); ++i)
        {
            auto code = program_pop.bytecode.ptr[i];
            auto code_length = program_pop.num_of_instructions.ptr[i];
            auto stack_length = program_pop.stack_req.ptr[i];
            auto intermediate_length = program_pop.intermediate_req.ptr[i];

            // Allocate memory for stack and intermediate
            stack_d.resize(stack_length, dataset->m);
            intermediate_d.resize(intermediate_length, dataset->m);

            // If the expression has no weights yet, initialize them randomly
            if (population[i].weights.empty()) {
                for (int j = 0; j < nweights; ++j) {
                    weights_d.ptr[j] = 2 * (thread_local_rng() % RAND_MAX) / (float)RAND_MAX - 1;
                }
            } 
            // If the expression already has weights, use them
            else {
                for (int j = 0; j < nweights; ++j) {
                    weights_d.ptr[j] = population[i].weights[j];
                }
            }

            // Do the work
            run(code, code_length, stack_length, intermediate_length, epochs, learning_rate);

            // Compute total loss
            float total_loss = 0;
            #pragma omp simd reduction(+:total_loss)
            for (int i = 0; i < dataset->m; ++i) {
                total_loss += loss_d.ptr[i];
            }

            // Save loss value to the original expression
            population[i].loss = total_loss;

            // Save weights to the original expression
            population[i].weights.resize(weights_d.ptr.dim1);
            for (int j = 0; j < weights_d.ptr.dim1; ++j) {
                population[i].weights[j] = weights_d.ptr[j];
            }
        }
    }
}