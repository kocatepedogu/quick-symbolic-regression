// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include <hip/hip_runtime.h>

#include "vm.hpp"
#include "vm_functions.hpp"
#include "vm_gradients.hpp"
#include "vm_debug.hpp"
#include "vm_propagation.hpp"

#include "../util.hpp"

constexpr int max_stack_depth = 1024;

template <PropagationType propType> __global__
void propagate_kernel(const Instruction* bytecode, int bytecode_length,
                      int m, const float *const *X_d, float *y_predicted_d,
                      float **stack_d, float **stack_intermediate_d) 
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < m; i += stride) {
        int stack_pointer = 0;
        int stack_intermediate_pointer = 0;

        const StackState s(
            stack_d,
            stack_intermediate_d,
            stack_pointer,
            stack_intermediate_pointer
        );

        for (int program_counter = 0; program_counter < bytecode_length; ++program_counter) {
            Instruction instruction = bytecode[program_counter];
            switch (instruction.opcode) {
                /* Operations with immediate operands */
                case PUSH_IMMEDIATE:
                    propagate_immediate<propType>(tid, instruction.value, s);
                    break;

                /* Operations with index operands */
                case PUSH_VARIABLE:
                    propagate_immediate<propType>(tid, X_d[instruction.argindex][tid], s);
                    break;
                case PUSH_PARAMETER:
                    break;

                /* Binary Operations */
                case ADD:
                    vm_debug_print("add");
                    propagate_binary<propType>(tid, forward_add, grad_add, s);
                    break;
                case SUB:
                    vm_debug_print("sub");
                    propagate_binary<propType>(tid, forward_sub, grad_sub, s);
                    break;
                case MUL:
                    vm_debug_print("mul");
                    propagate_binary<propType>(tid, forward_mul, grad_mul, s);
                    break;
                case DIV:
                    vm_debug_print("div");
                    propagate_binary<propType>(tid, forward_div, grad_div, s);
                    break;

                /* Unary Operations */
                case SIN:
                    vm_debug_print("sin");
                    propagate_unary<propType>(tid, forward_sin, grad_sin, s);
                    break;
                case COS:
                    vm_debug_print("cos");
                    propagate_unary<propType>(tid, forward_cos, grad_cos, s);
                    break;
                case EXP:
                    vm_debug_print("exp");
                    propagate_unary<propType>(tid, forward_exp, grad_exp, s);
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
    hipLaunchKernelGGL(propagate_kernel<FORWARD>, gridDim, blockDim, 0, 0,
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