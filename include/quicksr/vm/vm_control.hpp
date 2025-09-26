// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef INTRA_VM_CONTROL_HPP
#define INTRA_VM_CONTROL_HPP

#include <hip/hip_runtime.h>

#include "vm_functions.hpp"
#include "vm_gradients.hpp"
#include "vm_debug.hpp"
#include "vm_propagation.hpp"
#include "vm_types.hpp"

namespace qsr {

template <PropagationType propType, ParallelismType paraType, typename Weights, typename Code, typename... Debug> __device__ __host__
void vm_control(const ControlState<Code> c, const DataState &d, const StackState &s, const WeightState<Weights> &w, Debug ...debug)
{
    bool exit = false;

    for (; !exit && c.program_counter < c.bytecode_length; ++c.program_counter) {
        Instruction instruction;

        if constexpr (paraType == INTRA_INDIVIDUAL) {
            instruction = c.bytecode[c.program_counter];
        }
        if constexpr (paraType == INTER_INDIVIDUAL) {
            instruction = c.bytecode[c.program_counter,c.tid];
        }
        if constexpr (paraType == HYBRID) {
            instruction = c.bytecode[c.program_counter,c.tid / 16];
        }
        
        switch (instruction.opcode) {
            /* Operations with immediate operands */
            case PUSH_IMMEDIATE:
                vm_debug_print(c.tid, "imm %f", instruction.value);
                propagate_immediate<propType>(c.tid, instruction.value, s, debug...);
                break;

            /* Operations with index operands */
            case PUSH_VARIABLE:
                vm_debug_print(c.tid, "var %d", instruction.argindex);
                propagate_immediate<propType>(c.tid, d.X_d[instruction.argindex,c.datapoint_idx], s, debug...);
                break;
            case PUSH_PARAMETER:
                vm_debug_print(c.tid, "param %d", instruction.argindex);
                propagate_parameter<propType, paraType, Weights>(c.tid, instruction.argindex, s, w, debug...);
                break;

            /* Binary Operations */
            case ADD:
                vm_debug_print(c.tid, "add");
                propagate<propType, 2>(c.tid, forward_add, grad_add, s, instruction, debug...);
                break;
            case SUB:
                vm_debug_print(c.tid, "sub");
                propagate<propType, 2>(c.tid, forward_sub, grad_sub, s, instruction, debug...);
                break;
            case MUL:
                vm_debug_print(c.tid, "mul");
                propagate<propType, 2>(c.tid, forward_mul, grad_mul, s, instruction, debug...);
                break;
            case DIV:
                vm_debug_print(c.tid, "div");
                propagate<propType, 2>(c.tid, forward_div, grad_div, s, instruction, debug...);
                break;

            /* Unary Operations */
            case SIN:
                vm_debug_print(c.tid, "sin");
                propagate<propType, 1>(c.tid, forward_sin, grad_sin, s, instruction, debug...);
                break;
            case COS:
                vm_debug_print(c.tid, "cos");
                propagate<propType, 1>(c.tid, forward_cos, grad_cos, s, instruction, debug...);
                break;
            case EXP:
                vm_debug_print(c.tid, "exp");
                propagate<propType, 1>(c.tid, forward_exp, grad_exp, s, instruction, debug...);
                break;
            case RELU:
                vm_debug_print(c.tid, "relu");
                propagate<propType, 1>(c.tid, forward_relu, grad_relu, s, instruction, debug...);
                break;

            /* No operation */
            case NOP:
                vm_debug_print(c.tid, "nop");
                break;

            /* Loss function evaluation */
            case LOSS:
                vm_debug_print(c.tid, "loss");

                // Calculate loss and replace the value on the stack with the loss

                real& stack_value = s.stack_d[0,c.tid];

                const real y_predicted = stack_value;
                const real y_target = d.y_d[c.datapoint_idx];

                stack_value = (y_predicted - y_target) / (real)d.y_d.dim1;

                // Forward propagation has finished.

                exit = true;
                break;
        }

        vm_debug_print_stack(c.tid, s);
    }

    if constexpr (propType == FORWARD) {
        if (!exit) {
            assert(false && "Compiled program finished without encountering loss function.");
        }
    }
}

}

#endif
