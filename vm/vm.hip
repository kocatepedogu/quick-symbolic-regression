// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include <hip/hip_runtime.h>

#include "vm.hpp"
#include "../util.hpp"

constexpr bool vm_debug_messages = true;
constexpr int vm_debug_tid = 2;

constexpr int max_stack_depth = 1024;

template <typename... T>
__device__ static inline void vm_debug_print(T ...args) {
    if constexpr (vm_debug_messages) {
        if (blockDim.x * blockIdx.x + threadIdx.x == vm_debug_tid) {
            printf(args...);
            printf("\n");
        }
    }
}

__global__
void forward_propagate_kernel(const Instruction* bytecode, int bytecode_length,
                              int m, const float *const *X_d, float *y_predicted_d,
                              float **stack_d, float **stack_intermediate_d) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < m; i += stride) {
        int stack_pointer = 0;
        int stack_intermediate_pointer = 0;

        float operand1 = 0;
        float operand2 = 0; 

        for (int program_counter = 0; program_counter < bytecode_length; ++program_counter) {
            Instruction instruction = bytecode[program_counter];
            switch (instruction.opcode) {
                /* Operations with immediate operands */
                case PUSH_IMMEDIATE:
                    operand1 = instruction.value;
                    vm_debug_print("push %f", operand1);
                    stack_d[stack_pointer++][tid] = operand1;
                    break;

                /* Operations with index operands */
                case PUSH_VARIABLE:
                    operand1 = X_d[instruction.argindex][tid];
                    vm_debug_print("push %f", operand1);
                    stack_d[stack_pointer++][tid] = operand1;
                    break;
                case PUSH_PARAMETER:
                    break;

                /* Binary Operations */
                case ADD:
                    vm_debug_print("add");

                    operand1 = stack_d[--stack_pointer][tid];
                    operand2 = stack_d[--stack_pointer][tid];
                    stack_d[stack_pointer++][tid] = operand2 + operand1;
                    stack_intermediate_d[stack_intermediate_pointer++][tid] = operand1;
                    stack_intermediate_d[stack_intermediate_pointer++][tid] = operand2;

                    break;
                case SUB:
                    vm_debug_print("sub");

                    operand1 = stack_d[--stack_pointer][tid];
                    operand2 = stack_d[--stack_pointer][tid];
                    stack_d[stack_pointer++][tid] = operand2 - operand1;
                    stack_intermediate_d[stack_intermediate_pointer++][tid] = operand1;
                    stack_intermediate_d[stack_intermediate_pointer++][tid] = operand2;

                    break;
                case MUL:
                    vm_debug_print("mul");

                    operand1 = stack_d[--stack_pointer][tid];
                    operand2 = stack_d[--stack_pointer][tid];
                    stack_d[stack_pointer++][tid] = operand2 * operand1;
                    stack_intermediate_d[stack_intermediate_pointer++][tid] = operand1;
                    stack_intermediate_d[stack_intermediate_pointer++][tid] = operand2;

                    break;
                case DIV:
                    vm_debug_print("div");

                    operand1 = stack_d[--stack_pointer][tid];
                    operand2 = stack_d[--stack_pointer][tid];
                    stack_d[stack_pointer++][tid] = operand2 / operand1;
                    stack_intermediate_d[stack_intermediate_pointer++][tid] = operand1;
                    stack_intermediate_d[stack_intermediate_pointer++][tid] = operand2;

                    break;

                /* Unary Operations */
                case SIN:
                    vm_debug_print("sin");

                    operand1 = stack_d[--stack_pointer][tid];
                    stack_d[stack_pointer++][tid] = sin(operand1);
                    stack_intermediate_d[stack_intermediate_pointer++][tid] = operand1;

                    break;
                case COS:
                    vm_debug_print("cos");

                    operand1 = stack_d[--stack_pointer][tid];
                    stack_d[stack_pointer++][tid] = cos(operand1);
                    stack_intermediate_d[stack_intermediate_pointer++][tid] = operand1;

                    break;
                case EXP:
                    vm_debug_print("exp");

                    operand1 = stack_d[--stack_pointer][tid];
                    stack_d[stack_pointer++][tid] = exp(operand1);
                    stack_intermediate_d[stack_intermediate_pointer++][tid] = operand1;

                    break;

                /* No operation */
                case NOP:
                    vm_debug_print("nop");
                    break;

                /* Loss function evaluation */
                case LOSS:
                    vm_debug_print("loss");

                    // Save result
                    y_predicted_d[tid] = stack_d[0][tid];

                    // Print an additional newline
                    vm_debug_print("");

                    return;
            }
        }

        assert(false && "Compiled program finished without encountering loss function.");
    }
}

void forward_propagate(const Program& program, const Dataset& dataset) {
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

    /* Allocate array y_predicted_d for writing output values */ 
    float *y_predicted_d;
    HIP_CALL(hipMallocManaged(&y_predicted_d, sizeof *y_predicted_d * dataset.m));

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
     * Allocate array stack_intermediate_d to store intermediate calculation 
     * results for later use in backpropagation. The dimensions of 
     * stack_intermediate_d is the same as stack_d.
     */
    float **stack_intermediate_d;
    HIP_CALL(hipMallocManaged(&stack_intermediate_d, sizeof *stack_intermediate_d * max_stack_depth));
    for (int i = 0; i < max_stack_depth; ++i) {
        HIP_CALL(hipMallocManaged(&stack_intermediate_d[i], sizeof **stack_intermediate_d * dataset.m));
    }

    // Forward propagate
    hipLaunchKernelGGL(forward_propagate_kernel, gridDim, blockDim, 0, 0,
                       program.bytecode, 
                       program.length, 
                       dataset.m, 
                       dataset.X_d, 
                       y_predicted_d, 
                       stack_d,
                       stack_intermediate_d);

    HIP_CALL(hipDeviceSynchronize());

    // Print ground truth target values and predicted target values
    for (int i = 0; i < dataset.m; ++i) {
        std::cout << y_predicted_d[i] << ", " << dataset.y_d[i] << std::endl;
    }

    std::cout << std::endl;

    // Print intermediate calculation results for thread 2
    for (int i = 0; i < 20; ++i) {
        std::cout << stack_intermediate_d[i][2] << std::endl;
    }
}