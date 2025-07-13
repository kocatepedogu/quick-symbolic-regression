#include <hip/hip_runtime.h>
#include "vm_debug.hpp"
#include "vm_types.hpp"

#include "../compiler/bytecode.hpp"


template <PropagationType proptype> __device__
static inline void propagate_immediate(int tid, const float& value, const StackState& s) {
    if constexpr (proptype == FORWARD) {
        const float operand1 = value;
        s.stack_d[s.stack_pointer++][tid] = operand1;
    }

    if constexpr (proptype == BACK) {
        // Pop incoming gradient from the stack
        const float incoming_grad = s.stack_d[--s.stack_pointer][tid];
        vm_debug_print(tid, "  incoming_grad=%f", incoming_grad);
    }
}


template <PropagationType proptype, typename F, typename G> __device__ 
static inline void propagate_binary(int tid, F operation, G gradient, const StackState& s, const Instruction& inst) {
    if constexpr (proptype == FORWARD) {
        // Pop operands from the stack
        const float operand1 = s.stack_d[--s.stack_pointer][tid];
        const float operand2 = s.stack_d[--s.stack_pointer][tid];

        // Apply operation and push the result to the stack
        s.stack_d[s.stack_pointer++][tid] = operation(operand2, operand1);

        // Save consumed operands as intermediate values for backpropagation
        s.intermediate_d[s.intermediate_pointer++][tid] = operand2;
        s.intermediate_d[s.intermediate_pointer++][tid] = operand1;
    }

    if constexpr (proptype == BACK) {
        // Pop incoming gradient from the stack
        const float incoming_grad = s.stack_d[--s.stack_pointer][tid];
        vm_debug_print(tid, "  incoming_grad=%f", incoming_grad);

        // Pop intermediate calculation results from forward propagation from intermediate_d
        const float intermediate1 = s.intermediate_d[inst.intermediate_index + 0][tid];
        const float intermediate2 = s.intermediate_d[inst.intermediate_index + 1][tid];
        vm_debug_print(tid, "  intermediate1=%f", intermediate1);
        vm_debug_print(tid, "  intermediate2=%f", intermediate2);

        // Calculate local gradients
        float grad1;
        float grad2;
        gradient(intermediate1, intermediate2, grad1, grad2);
        vm_debug_print(tid, "  grad1=%f", grad1);
        vm_debug_print(tid, "  grad2=%f", grad2);

        // Calculate outgoing gradients
        grad1 *= incoming_grad;
        grad2 *= incoming_grad;

        // Push outgoing gradients to the stack
        s.stack_d[s.stack_pointer++][tid] = grad2;
        s.stack_d[s.stack_pointer++][tid] = grad1;
    }
}


template <PropagationType proptype, typename F, typename G> __device__ 
static inline void propagate_unary(int tid, F operation, G gradient, const StackState& s, const Instruction& inst) {
    if constexpr (proptype == FORWARD) {
        // Pop operand from the stack
        const float operand1 = s.stack_d[--s.stack_pointer][tid];

        // Apply operation and push the result to the stack
        s.stack_d[s.stack_pointer++][tid] = operation(operand1);

        // Save consumed operand as intermediate value for backpropagation
        s.intermediate_d[s.intermediate_pointer++][tid] = operand1;
    }

    if constexpr (proptype == BACK) {
        // Pop incoming gradient from the stack
        const float incoming_grad = s.stack_d[--s.stack_pointer][tid];
        vm_debug_print(tid, "  incoming_grad=%f", incoming_grad);

        // Pop intermediate calculation result from forward propagation from intermediate_d
        const float intermediate1 = s.intermediate_d[inst.intermediate_index][tid];
        vm_debug_print(tid, "  intermediate1=%f", intermediate1);

        // Calculate local gradients
        float grad1;
        gradient(intermediate1, grad1);
        vm_debug_print(tid, "  grad1=%f", grad1);

        // Calculate outgoing gradients
        grad1 *= incoming_grad;

        // Push outgoing gradients to the stack
        s.stack_d[s.stack_pointer++][tid] = grad1;
    }
}
