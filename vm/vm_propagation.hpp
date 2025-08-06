// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef INTRA_VM_PROPAGATION_HPP
#define INTRA_VM_PROPAGATION_HPP

#include <hip/hip_runtime.h>

#include "vm_debug.hpp"
#include "vm_types.hpp"

#include "../compiler/ir.hpp"

namespace qsr {

__device__ __host__
static inline void push_stack(const StackState &s, int tid, float value) {
    s.stack_d[s.stack_pointer++,tid] = value;
}

__device__ __host__
static inline float pop_stack(const StackState &s, int tid) {
    return s.stack_d[--s.stack_pointer,tid];
}

__device__ __host__
static inline void push_intermediate(const StackState &s, int tid, float value) {
    s.intermediate_d[s.intermediate_pointer++,tid] = value;
}

__device__ __host__
static inline float read_intermediate(const StackState &s, int tid, int index) {
    return s.intermediate_d[index,tid];
}

template <PropagationType proptype> __device__ __host__
static inline void propagate_immediate(int tid, const float& immediate, const StackState& s) {
    if constexpr (proptype == FORWARD) {
        push_stack(s, tid, immediate);
    }

    if constexpr (proptype == BACK) {
        // Pop gradient from stack, discard the value.
        pop_stack(s, tid);
    }
}

template <PropagationType proptype, ParallelismType paraType, typename Weight> __device__ __host__
static inline void propagate_parameter(int tid, const int& param_index, const StackState& s,
                                    Weight weights, 
                                    Ptr2D<float> weights_grad_d) {
    if constexpr (proptype == FORWARD) {
        if constexpr (paraType == INTRA_INDIVIDUAL) {
            // Intra-individual
            push_stack(s, tid, weights[param_index]);
        }
        else {
            /// Inter-individual
            push_stack(s, tid, weights[param_index,tid]);
        }
    }

    if constexpr (proptype == BACK) {
        // Pop gradient from stack
        const float incoming_grad = pop_stack(s, tid);
        vm_debug_print(tid, "  incoming_grad=%f", incoming_grad);

        // Add gradient to the total gradient of the associated trainable parameter
        weights_grad_d[param_index,tid] += incoming_grad;
    }
}

template <PropagationType proptype, int opcount, typename F, typename G> __device__ __host__
static inline void propagate(int tid, F operation, G gradient, const StackState& s, const Instruction& inst) {
    if constexpr (proptype == FORWARD) {
        // Pop operands from the stack
        float operands[opcount];
        #pragma unroll
        for (int k = opcount - 1; k >= 0; k--) {
            operands[k] = pop_stack(s, tid);
        }

        // Apply operation and push the result to the stack
        push_stack(s, tid, operation(operands));

        // Save consumed operand as intermediate value for backpropagation
        #pragma unroll
        for (int k = 0; k < opcount; ++k) {
            push_intermediate(s, tid, operands[k]);
        }
    }

    if constexpr (proptype == BACK) {
        // Pop incoming gradient from the stack
        const float incoming_grad = pop_stack(s, tid);
        vm_debug_print(tid, "  incoming_grad=%f", incoming_grad);

        // Pop intermediate calculation results from forward propagation from intermediate_d
        float intermediates[opcount];
        #pragma unroll
        for (int k = opcount - 1; k >= 0; k--) {
            const float intermediate = read_intermediate(s, tid, inst.intermediate_index + k);
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
            push_stack(s, tid, local_grad[k]);
        }
    }
}

}

#endif
