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

template <PropagationType propType> __device__
void vm_control(const int tid, const Instruction* bytecode, int bytecode_length,
                int m, const float *const *X_d, const float *y_d, float *loss_d,
                const StackState& s, int& program_counter) 
{
    for (; program_counter < bytecode_length; ++program_counter) {
        Instruction instruction = bytecode[program_counter];
        switch (instruction.opcode) {
            /* Operations with immediate operands */
            case PUSH_IMMEDIATE:
                vm_debug_print("push immediate %f", instruction.value);
                propagate_immediate<propType>(tid, instruction.value, s);
                break;

            /* Operations with index operands */
            case PUSH_VARIABLE:
                vm_debug_print("push variable %d", instruction.argindex);
                propagate_immediate<propType>(tid, X_d[instruction.argindex][tid], s);
                break;
            case PUSH_PARAMETER:
                vm_debug_print("push parameter %d", instruction.argindex);
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

                // Save loss
                const float y_predicted = s.stack_d[0][tid];
                const float y_target = y_d[tid];
                loss_d[tid] = 1; //y_target - y_predicted;

                // Print an additional newline
                vm_debug_print("");

                return;
        }
    }

    if constexpr (propType == FORWARD) {
        assert(false && "Compiled program finished without encountering loss function.");
    }
}

__global__
void vm(const Instruction* bytecode, int bytecode_length,
        int m, const float *const *X_d, const float *y_d, float *loss_d,
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

        int program_counter = 0;

        // Forward propagate and evaluate loss
        vm_control<FORWARD>(tid, bytecode, bytecode_length, m, X_d, y_d, loss_d, s, program_counter);

        // Reset stack, push loss to the stack as the first element
        stack_pointer = 1;
        stack_d[0][tid] = loss_d[tid];

        // Backpropagate
        vm_control<BACK>(tid, bytecode, bytecode_length, m, X_d, y_d, loss_d, s, ++program_counter);
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
    hipLaunchKernelGGL(vm, gridDim, blockDim, 0, 0,
                       program.bytecode, 
                       program.length, 
                       dataset.m, 
                       dataset.X_d, 
                       dataset.y_d,
                       loss_d, 
                       stack_d,
                       stack_intermediate_d);

    HIP_CALL(hipDeviceSynchronize());

    // Print loss for each data point
    for (int i = 0; i < dataset.m; ++i) {
        std::cout << "Loss " << i << " " << loss_d[i] << std::endl;
    } std::cout << std::endl;

    // Print intermediate calculation results for thread 2
    for (int i = 0; i < 20; ++i) {
        std::cout << stack_intermediate_d[i][2] << std::endl;
    }
}