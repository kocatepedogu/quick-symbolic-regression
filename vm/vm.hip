// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

#include </usr/lib/clang/20/include/omp.h>

#include "vm.hpp"
#include "vm_debug.hpp"
#include "vm_control.hpp"

#include "../util.hpp"

constexpr int max_stack_depth = 32;

__global__
void vm(const Instruction* bytecode, 
        const int bytecode_length,
        const int m, 
        const float *const *const X_d, 
        const float *const y_d, 
        float *const *const stack_d, 
        float *const *const intermediate_d,
        float *const weights_d,
        float *const *const weights_grad_d,
        const int nweights,
        float *const *const weights_grad_reduced_sum_d,
        const int epochs,
        const float learning_rate
    )
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    auto grid_group = cooperative_groups::this_grid();

    // Shared memory for block-level reduction
    __shared__ float sharedMem[1024];

    for (int iter = 0; iter < epochs; ++iter) {
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

        grid_group.sync();

        // Reduce sum individual gradients coming from each data point
        for (int weight_idx = 0; weight_idx < nweights; ++weight_idx) {
            // Load data into shared memory
            sharedMem[threadIdx.x] = (tid < m) ? weights_grad_d[weight_idx][tid] : 0.0f;
            __syncthreads();
            
            // Parallel reduction within the block
            for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) {
                    sharedMem[threadIdx.x] += sharedMem[threadIdx.x + stride];
                }
                __syncthreads();
            }
            
            // First thread of each block writes block sum to output
            if (threadIdx.x == 0) {
                atomicAdd(&weights_d[weight_idx], -learning_rate * sharedMem[0]);
            }
        }

        grid_group.sync();
    }    
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
            gridDim.x * sizeof **weights_grad_reduced_sum_d));
    }
}

void VirtualMachine::fit(const Program& program, int epochs, float learning_rate) {
    // Randomly initialize weights with values between -1.0 and 1.0
    for (int i = 0; i < nweights; ++i) {
        weights_d[i] = 0.5; //2.0 * rand() / (float)RAND_MAX - 1.0;
    }

    const void *params[] = {
        &program.bytecode, &program.length, 
        &dataset.m, &dataset.X_d, &dataset.y_d,
        &stack_d, &intermediate_d,
        &weights_d, &weights_grad_d, &nweights, &weights_grad_reduced_sum_d,
        &epochs, &learning_rate
    };

    HIP_CALL(hipLaunchCooperativeKernel(
        vm, gridDim, blockDim, 
        (void **)params, 0, stream));

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