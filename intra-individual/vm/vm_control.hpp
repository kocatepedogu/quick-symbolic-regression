// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef INTRA_VM_CONTROL_HPP
#define INTRA_VM_CONTROL_HPP

#include <hip/hip_runtime.h>

#include "../../vm/vm_functions.hpp"
#include "../../vm/vm_gradients.hpp"
#include "../../vm/vm_debug.hpp"
#include "vm_propagation.hpp"

namespace intra_individual {
    template <PropagationType propType, ParallelismType paraType> __device__
    void vm_control(const int tid, const int datapoint_idx,
                    const Instruction* bytecode, 
                    const int bytecode_length,
                    const int m, 
                    const float *const __restrict__ *const __restrict__ X_d, 
                    const float *const __restrict__ y_d,
                    const StackState& s, 
                    int& program_counter,
                    float *const __restrict__ weights_d,
                    float *const __restrict__ *const __restrict__ weights_grad_d)
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
                    if constexpr (paraType == INTRA_INDIVIDUAL) {
                        propagate_immediate<propType>(tid, X_d[instruction.argindex][tid], s);
                    }
                    if constexpr (paraType == INTER_INDIVIDUAL) {
                        propagate_immediate<propType>(tid, X_d[datapoint_idx][instruction.argindex], s);
                    }
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

                    stack_value = (y_predicted - y_target) / m;

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
}

#endif
