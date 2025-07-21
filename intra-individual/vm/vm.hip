// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include <hip/hip_runtime.h>
#include </usr/lib/clang/20/include/omp.h>

#include "vm.hpp"
#include "vm_debug.hpp"
#include "vm_control.hpp"

#include "../../util.hpp"

namespace intra_individual {

    constexpr int max_stack_depth = 32;

    constexpr int reduction_threads_per_block = 32;

    __global__ 
    void weight_update(const float *const __restrict__ weight_grads, 
                    float *const __restrict__ weight, 
                    int m, float learning_rate) {
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

    __global__
    void vm(const Instruction *bytecode, 
            const int bytecode_length,
            const int m, 
            const float *const __restrict__ *const __restrict__ X_d, 
            const float *const __restrict__ y_d, 
            float *const __restrict__ *const __restrict__ stack_d, 
            float *const __restrict__ *const __restrict__ intermediate_d,
            float *const __restrict__ weights_d,
            float *const __restrict__ *const __restrict__ weights_grad_d,
            const int nweights)
    {
        const int tid = blockDim.x * blockIdx.x + threadIdx.x;

        // Reset weight gradients that will be computed by this thread
        for (int j = 0; j < nweights; ++j) {
            weights_grad_d[j][tid] = 0;
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
        vm_control<FORWARD>(tid, bytecode, bytecode_length, m, X_d, y_d, s, program_counter, weights_d, weights_grad_d);

        // Print an empty line in between forward propagation output and backpropagation output
        vm_debug_print(tid, "");

        // Backpropagate
        vm_debug_print(tid, "Backpropagation");
        vm_control<BACK>(tid, bytecode, bytecode_length, m, X_d, y_d, s, program_counter, weights_d, weights_grad_d);
    }

    VirtualMachine::VirtualMachine(const Dataset& dataset, hipStream_t& stream, int nweights, omp_lock_t& print_lock) :
        dataset(dataset), stream(stream), nweights(nweights), print_lock(print_lock)
    {
        HIP_CALL(hipGetDevice(&device_id));
        HIP_CALL(hipGetDeviceProperties(&props, device_id));

        /* 
        * Decide number of blocks and threads for computation
        * - The total number of threads must be greater than or equal to 
        *   the number of data points (batch size = m).
        * - Each block will have the maximum number of threads supported by
        *   the device (probably 1024).
        * - Excess threads are later masked inside kernel with if (tid < m).
        */

        const int threadsPerBlock = min(dataset.m, props.maxThreadsPerBlock);
        const int blocks = (dataset.m + threadsPerBlock - 1) / threadsPerBlock;

        gridDim = dim3(blocks);
        blockDim = dim3(threadsPerBlock);

        /* 
        * Decide number of blocks for reduce sum
        * - Each block will produce a separate block sum. These block sums are then
        *   sequentially summed to produce the final sum for each weight.
        */

        int reduction_blocks_per_grid = (dataset.m + reduction_threads_per_block - 1) / 
            reduction_threads_per_block;

        reduction_grid_dim = dim3(reduction_blocks_per_grid);
        reduction_block_dim = dim3(reduction_threads_per_block);

        /* 
        * Allocate array stack_d as stack memory for bytecode virtual machine.
        * - Each thread accesses its own stack. 
        * - The stack pointer is the same for each thread at any given point in time.
        *   There are no instructions that can lead to divergent control flow.
        * - Consecutive threads should access consecutive locations in the stack.
        *   The dimensions of the stack are [max_stack_depth][num_threads]
        */
        HIP_CALL(hipMallocManaged(&stack_d, sizeof *stack_d * max_stack_depth));
        for (int i = 0; i < max_stack_depth; ++i) {
            HIP_CALL(hipMallocManaged(&stack_d[i], sizeof **stack_d * dataset.m));
        }

        /*
        * Allocate array intermediate_d to store intermediate calculation 
        * results for later use in backpropagation. The dimensions of 
        * intermediate_d is the same as stack_d.
        */
        HIP_CALL(hipMallocManaged(&intermediate_d, sizeof *intermediate_d * max_stack_depth));
        for (int i = 0; i < max_stack_depth; ++i) {
            HIP_CALL(hipMallocManaged(&intermediate_d[i], sizeof **intermediate_d * dataset.m));
        }

        /* Allocate weights */
        HIP_CALL(hipMallocManaged(&weights_d, sizeof *weights_d * nweights));

        /*
        * Allocate array weights_grad_d.
        * - For each weight, there is an array of gradients, whose elements are gradients 
        *   computed from different data points.
        * - weights_grad_d is an array of arrays with dimensions [nweights][num_threads]
        */
        HIP_CALL(hipMallocManaged(&weights_grad_d, sizeof *weights_grad_d * nweights));
        for (int i = 0; i < nweights; ++i) {
            HIP_CALL(hipMallocManaged(&weights_grad_d[i], sizeof **weights_grad_d * dataset.m));
        }

        // Allocate memory for block sums
        HIP_CALL(hipMallocManaged(&weights_grad_reduced_sum_d, 
            nweights * sizeof *weights_grad_reduced_sum_d));
        for (int i = 0; i < nweights; ++i) {
            HIP_CALL(hipMallocManaged(&weights_grad_reduced_sum_d[i], 
                reduction_blocks_per_grid * sizeof **weights_grad_reduced_sum_d));
        }
    }

    void VirtualMachine::fit(const Instruction* code, int code_length, int epochs, float learning_rate) {
        // Randomly initialize weights with values between -1.0 and 1.0
        for (int i = 0; i < nweights; ++i) {
            weights_d[i] = 0.5; //2.0 * rand() / (float)RAND_MAX - 1.0;
        }

        for (int i = 0; i < epochs; ++i) {
            // Run forward propagation and backpropagation to compute
            // weight gradients for each weight and for each data point
            hipLaunchKernelGGL(
                vm, gridDim, blockDim, 0, stream,
                code, code_length, 
                dataset.m, dataset.X_d, dataset.y_d,
                stack_d, intermediate_d,
                weights_d, weights_grad_d, nweights);

            // For every weight, sum gradient contributions from all data points using reduction
            // Apply gradient descent rule
            for (int i = 0; i < nweights; ++i) {
                hipLaunchKernelGGL(
                    weight_update, reduction_grid_dim, reduction_block_dim, 0, stream, 
                    weights_grad_d[i], &weights_d[i], dataset.m, learning_rate);
            }
        }

        HIP_CALL(hipStreamSynchronize(stream));

        // Print calculated weights
        omp_set_lock(&print_lock);
        for (int i = 0; i < nweights; ++i) {
            std::cout << weights_d[i] << std::endl;
        } std::cout << std::endl;
        omp_unset_lock(&print_lock);
    }

    VirtualMachine::~VirtualMachine() {
        // Deallocate array for block sums
        for (int i = 0; i < nweights; ++i) {
            HIP_CALL(hipFree(weights_grad_reduced_sum_d[i]));
        }
        HIP_CALL(hipFree(weights_grad_reduced_sum_d));

        // Deallocate array weights_grad_d.
        for (int i = 0; i < nweights; ++i) {
            HIP_CALL(hipFree(weights_grad_d[i]));
        }
        HIP_CALL(hipFree(weights_grad_d));

        // Deallocate array weights_d
        HIP_CALL(hipFree(weights_d));

        // Deallocate array intermediate_d.
        for (int i = 0; i < max_stack_depth; ++i) {
            HIP_CALL(hipFree(intermediate_d[i]));
        }
        HIP_CALL(hipFree(intermediate_d));

        // Deallocate array stack_d
        for (int i = 0; i < max_stack_depth; ++i) {
            HIP_CALL(hipFree(stack_d[i]));
        }
        HIP_CALL(hipFree(stack_d));
    }
}