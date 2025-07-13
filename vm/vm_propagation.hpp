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

template <PropagationType proptype, int opcount, typename F, typename G> __device__ 
static inline void propagate(int tid, F operation, G gradient, const StackState& s, const Instruction& inst) {
    if constexpr (proptype == FORWARD) {
        // Pop operands from the stack
        float operands[opcount];
        #pragma unroll
        for (int k = opcount - 1; k >= 0; k--) {
            operands[k] = s.stack_d[--s.stack_pointer][tid];
        }

        // Apply operation and push the result to the stack
        s.stack_d[s.stack_pointer++][tid] = operation(operands);

        // Save consumed operand as intermediate value for backpropagation
        #pragma unroll
        for (int k = 0; k < opcount; ++k) {
            s.intermediate_d[s.intermediate_pointer++][tid] = operands[k];
        }
    }

    if constexpr (proptype == BACK) {
        // Pop incoming gradient from the stack
        const float incoming_grad = s.stack_d[--s.stack_pointer][tid];
        vm_debug_print(tid, "  incoming_grad=%f", incoming_grad);

        // Pop intermediate calculation results from forward propagation from intermediate_d
        float intermediates[opcount];
        #pragma unroll
        for (int k = opcount - 1; k >= 0; k--) {
            const float intermediate = s.intermediate_d[inst.intermediate_index + k][tid];
            intermediates[k] = intermediate;
            vm_debug_print(tid, "  intermediates[%d]=%f", k, intermediate);
        }

        // Calculate local gradients
        float local_grad[opcount];
        gradient(intermediates, local_grad);
        #pragma unroll
        for (int k = opcount - 1; k >= 0; k--) {
            vm_debug_print(tid, "  local_grad[%d]=%f", k, local_grad[k]);
        }

        // Calculate outgoing gradients
        #pragma unroll
        for (int k = opcount - 1; k >= 0; k--) {
            local_grad[k] *= incoming_grad;
        }

        // Push outgoing gradients to the stack
        #pragma unroll
        for (int k = opcount - 1; k >= 0; k--) {
            s.stack_d[s.stack_pointer++][tid] = local_grad[k];
        }
    }
}