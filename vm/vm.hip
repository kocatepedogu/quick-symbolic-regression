// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include <hip/hip_runtime.h>

#include "vm.hpp"
#include "vm_functions.hpp"
#include "vm_gradients.hpp"
#include "vm_debug.hpp"
#include "vm_propagation.hpp"
#include "vm_helper.hpp"

#include "../util.hpp"

constexpr int max_stack_depth = 1024;

void gradient_descent(float *const *const weights_grad_d, float *weights_d, int m, int nweights) {
    /* 
     * The individual gradients from each data point will be summed using reduction.
     * There is a separate sum for every trainable weight.
     */

    // Threads per block for reduce sum
    int threadsPerBlock = reduction_threads_per_block;

    // Number of blocks for reduce sum
    // Each block will produce a separate block sum. These block sums are then
    // sequentially summed again to produce the final sum for each weight.
    int blocksPerGrid = (m + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate memory for block sums
    float **block_sums;
    HIP_CALL(hipMallocManaged(&block_sums, nweights * sizeof *block_sums));
    for (int i = 0; i < nweights; ++i) {
        HIP_CALL(hipMallocManaged(&block_sums[i], blocksPerGrid * sizeof **block_sums));
    }
    
    // Launch a separate reduction kernel for every weight
    for (int i = 0; i < nweights; ++i) {
        hipLaunchKernelGGL(reduce_sum, 
                        dim3(blocksPerGrid), 
                        dim3(threadsPerBlock), 
                        0, 0, 
                        weights_grad_d[i], block_sums[i], m);
    }

    // Wait for all reductions to finish
    HIP_CALL(hipDeviceSynchronize());
    
    // Compute final sum on host
    for (int i = 0; i < nweights; ++i) {
        float totalSum = 0.0f;
        for (int j = 0; j < blocksPerGrid; ++j) {
            totalSum += block_sums[i][j];
        }
        weights_d[i] -= 0.0001 * totalSum;
    }

    // Free memory for block sums
    for (int i = 0; i < nweights; ++i) {
        HIP_CALL(hipFree(block_sums[i]));
    }
    HIP_CALL(hipFree(block_sums));
}

template <PropagationType propType> __device__
void vm_control(const int tid, 
                const Instruction* bytecode, 
                const int bytecode_length,
                const int m, 
                const float *const *const X_d, 
                const float *const y_d,
                const StackState& s, 
                int& program_counter,
                float *const weights_d,
                float *const *const weights_grad_d)
{
    bool exit = false;

    for (; !exit && program_counter < bytecode_length; ++program_counter) {
        Instruction instruction = bytecode[program_counter];
        switch (instruction.opcode) {
            /* Operations with immediate operands */
            case PUSH_IMMEDIATE:
                vm_debug_print(tid, "imm %f", instruction.value);
                propagate_immediate<propType>(tid, instruction.value, s);
                break;

            /* Operations with index operands */
            case PUSH_VARIABLE:
                vm_debug_print(tid, "var %d", instruction.argindex);
                propagate_variable<propType>(tid, instruction.argindex, s, 
                    X_d);
                break;
            case PUSH_PARAMETER:
                vm_debug_print(tid, "param %d", instruction.argindex);
                propagate_parameter<propType>(tid, instruction.argindex, s, 
                    weights_d, weights_grad_d);
                break;

            /* Binary Operations */
            case ADD:
                vm_debug_print(tid, "add");
                propagate<propType, 2>(tid, forward_add, grad_add, s, instruction);
                break;
            case SUB:
                vm_debug_print(tid, "sub");
                propagate<propType, 2>(tid, forward_sub, grad_sub, s, instruction);
                break;
            case MUL:
                vm_debug_print(tid, "mul");
                propagate<propType, 2>(tid, forward_mul, grad_mul, s, instruction);
                break;
            case DIV:
                vm_debug_print(tid, "div");
                propagate<propType, 2>(tid, forward_div, grad_div, s, instruction);
                break;

            /* Unary Operations */
            case SIN:
                vm_debug_print(tid, "sin");
                propagate<propType, 1>(tid, forward_sin, grad_sin, s, instruction);
                break;
            case COS:
                vm_debug_print(tid, "cos");
                propagate<propType, 1>(tid, forward_cos, grad_cos, s, instruction);
                break;
            case EXP:
                vm_debug_print(tid, "exp");
                propagate<propType, 1>(tid, forward_exp, grad_exp, s, instruction);
                break;

            /* No operation */
            case NOP:
                vm_debug_print(tid, "nop");
                break;

            /* Loss function evaluation */
            case LOSS:
                vm_debug_print(tid, "loss");

                // Calculate loss and replace the value on the stack with the loss

                float& stack_value = s.stack_d[0][tid];

                const float y_predicted = stack_value;
                const float y_target = y_d[tid];

                stack_value = y_predicted - y_target;

                // Forward propagation has finished.

                exit = true;
                break;
        }

        vm_debug_print_stack(tid, s);
    }

    if constexpr (propType == FORWARD) {
        if (!exit) {
            assert(false && "Compiled program finished without encountering loss function.");
        }
    }
}

__global__
void vm(const Instruction* bytecode, 
        const int bytecode_length,
        const int m, 
        const float *const *const X_d, 
        const float *const y_d, 
        const float *const loss_d,
        float *const *const stack_d, 
        float *const *const intermediate_d,
        float *const weights_d,
        float *const *const weights_grad_d)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < m; i += stride) {
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
}

void fit(const Program& program, const Dataset& dataset, int nweights) {
    /* 
     * Decide number of blocks and threads 
     * - The total number of threads must be greater than or equal to 
     *   the number of data points.
     * - Each block will have the maximum number of threads supported by
     *   the device (probably 1024).
     * - Excess threads are later masked inside kernel with if (tid < m).
     */ 

    int deviceId;
    HIP_CALL(hipGetDevice(&deviceId));

    hipDeviceProp_t props;
    HIP_CALL(hipGetDeviceProperties(&props, deviceId));
    int maxThreadsPerBlock = props.maxThreadsPerBlock;

    int threadsPerBlock = maxThreadsPerBlock;
    int blocks = (dataset.m + threadsPerBlock - 1) / threadsPerBlock;

    dim3 gridDim(blocks);
    dim3 blockDim(threadsPerBlock);

    /* Allocate array loss_d for writing loss values */ 
    float *loss_d;
    HIP_CALL(hipMallocManaged(&loss_d, sizeof *loss_d * dataset.m));

    /* 
     * Allocate array stack_d as stack memory for bytecode virtual machine.
     * - Each thread accesses its own stack. 
     * - The stack pointer is the same for each thread at any given point in time.
     *   There are no instructions that can lead to divergent control flow.
     * - Consecutive threads should access consecutive locations in the stack.
     *   The dimensions of the stack are [max_stack_depth][num_threads]
     */
    float **stack_d;
    HIP_CALL(hipMallocManaged(&stack_d, sizeof *stack_d * max_stack_depth));
    for (int i = 0; i < max_stack_depth; ++i) {
        HIP_CALL(hipMallocManaged(&stack_d[i], sizeof **stack_d * dataset.m));
    }

    /*
     * Allocate array intermediate_d to store intermediate calculation 
     * results for later use in backpropagation. The dimensions of 
     * intermediate_d is the same as stack_d.
     */
    float **intermediate_d;
    HIP_CALL(hipMallocManaged(&intermediate_d, sizeof *intermediate_d * max_stack_depth));
    for (int i = 0; i < max_stack_depth; ++i) {
        HIP_CALL(hipMallocManaged(&intermediate_d[i], sizeof **intermediate_d * dataset.m));
    }

    /* Allocate array weights_d for storing weights */
    float *weights_d;
    HIP_CALL(hipMallocManaged(&weights_d, sizeof *weights_d * nweights));

    /* Initialize weights with random values between -1.0 and 1.0 */
    for (int i = 0; i < nweights; ++i) {
        weights_d[i] = 2.0 * rand() / (float)RAND_MAX - 1.0;
    }

    /*
     * Allocate array weights_grad_d.
     * - For each weight, there is an array of gradients, whose elements are gradients 
     *   computed from different data points.
     * - weights_grad_d is an array of arrays with dimensions [nweights][num_threads]
     */
    float **weights_grad_d;
    HIP_CALL(hipMallocManaged(&weights_grad_d, sizeof *weights_grad_d * nweights));
    for (int i = 0; i < nweights; ++i) {
        HIP_CALL(hipMallocManaged(&weights_grad_d[i], sizeof **weights_grad_d * dataset.m));
    }

    std::cout << "Start" << std::endl;
    
    for (int i = 0; i < 3000; ++i) {
        /* Initialize weight gradients with zero */
        for (int i = 0; i < nweights; ++i) {
            for (int j = 0; j < dataset.m; ++j) {
                weights_grad_d[i][j] = 0.0;
            }
        }

        // Run forward propagation and backpropagation
        hipLaunchKernelGGL(vm, gridDim, blockDim, 0, 0,
                        program.bytecode, 
                        program.length, 
                        dataset.m, 
                        dataset.X_d, 
                        dataset.y_d,
                        loss_d, 
                        stack_d,
                        intermediate_d,
                        weights_d,
                        weights_grad_d);

        HIP_CALL(hipDeviceSynchronize());

        // Apply gradient descent weight updates
        gradient_descent(weights_grad_d, weights_d, dataset.m, nweights);
    }

    // Print calculated weights
    for (int i = 0; i < nweights; ++i) {
        std::cout << weights_d[i] << std::endl;
    }
}