// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef INTRA_VM_PROPAGATION_HPP
#define INTRA_VM_PROPAGATION_HPP

#include <hip/hip_runtime.h>

#include "vm_debug.hpp"
#include "vm_types.hpp"

#include "compiler/ir.hpp"

namespace qsr {

template <typename... Debug>
__device__ __host__
static inline void push_stack(const StackState &s, int tid, real value, Debug ...debug) {
    if constexpr (sizeof...(debug) == 2) {
        auto stack_length = std::get<0>(std::forward_as_tuple(debug...));
        if (s.stack_pointer >= stack_length) {
            printf("Error (push_stack): Stack buffer overflow at thread %d\n Stack pointer: %d, Stack length: %d\n", tid, s.stack_pointer, stack_length);
            abort();
        }
    }

    s.stack_d[s.stack_pointer++,tid] = value;
}

template <typename... Debug>
__device__ __host__
static inline real pop_stack(const StackState &s, int tid, Debug ...debug) {
    --s.stack_pointer;

    if constexpr (sizeof...(debug) == 2) {
        auto stack_length = std::get<0>(std::forward_as_tuple(debug...));
        if (s.stack_pointer >= stack_length) {
            printf("Error (pop_stack): Stack buffer overflow at thread %d\n Stack pointer: %d, Stack length: %d\n", tid, s.stack_pointer, stack_length);
            abort();
        }
    }

    return s.stack_d[s.stack_pointer,tid];
}

template <typename... Debug>
__device__ __host__
static inline void push_intermediate(const StackState &s, int tid, real value, Debug ...debug) {
    if constexpr (sizeof...(debug) == 2) {
        auto intermediate_length = std::get<1>(std::forward_as_tuple(debug...));
        if (s.intermediate_pointer >= intermediate_length) {
            printf("Error (push_intermediate): Intermediate buffer overflow at thread %d\n Intermediate pointer: %d, Intermediate length: %d\n", tid, s.intermediate_pointer, intermediate_length);
            abort();
        }
    }

    s.intermediate_d[s.intermediate_pointer++,tid] = value;
}

template <typename... Debug>
__device__ __host__
static inline real read_intermediate(const StackState &s, int tid, int index, Debug ...debug) {
    if constexpr (sizeof...(debug) == 2) {
        auto intermediate_length = std::get<1>(std::forward_as_tuple(debug...));
        if (index >= intermediate_length) {
            printf("Error (read_intermediate): Intermediate buffer overflow (index) at thread %d\n Index: %d, Intermediate length: %d\n", tid, index, intermediate_length);
            abort();
        }
    }

    return s.intermediate_d[index,tid];
}

template <PropagationType proptype, typename... Debug> __device__ __host__
static inline void propagate_immediate(int tid, const real& immediate, const StackState& s, Debug ...debug) {
    if constexpr (proptype == FORWARD) {
        push_stack(s, tid, immediate, debug...);
    }

    if constexpr (proptype == BACK) {
        // Pop gradient from stack, discard the value.
        pop_stack(s, tid, debug...);
    }
}

template <PropagationType proptype, ParallelismType paraType, typename Weight, typename... Debug> __device__ __host__
static inline void propagate_parameter(int tid, const int& param_index, const StackState& s,
                                       const WeightState<Weight> &w, Debug ...debug) {
    if constexpr (proptype == FORWARD) {
        if constexpr (paraType == INTRA_INDIVIDUAL) {
            push_stack(s, tid, w.weights_d[param_index], debug...);
        }
        if constexpr (paraType == INTER_INDIVIDUAL) {
            push_stack(s, tid, w.weights_d[param_index,tid], debug...);
        }
        if constexpr (paraType == HYBRID) {
            constexpr int datapoint_block_dim = 16;
            push_stack(s, tid, w.weights_d[param_index,tid / datapoint_block_dim], debug...);
        }
    }

    if constexpr (proptype == BACK) {
        // Pop gradient from stack
        const real incoming_grad = pop_stack(s, tid, debug...);
        vm_debug_print(tid, "  incoming_grad=%f", incoming_grad);

        // Add gradient to the total gradient of the associated trainable parameter
        if constexpr (paraType == INTRA_INDIVIDUAL || paraType == INTER_INDIVIDUAL) {
            w.weights_grad_d[param_index,tid] += incoming_grad;
        }

        if constexpr (paraType == HYBRID) {
            constexpr int datapoint_block_dim = 16;
            atomicAdd(&w.weights_grad_d[param_index,tid / datapoint_block_dim], incoming_grad);
        }
    }
}

template <PropagationType proptype, int opcount, typename F, typename G, typename... Debug> __device__ __host__
static inline void propagate(int tid, F operation, G gradient, const StackState& s, const Instruction& inst, Debug ...debug) {
    if constexpr (proptype == FORWARD) {
        // Pop operands from the stack
        real operands[opcount];
        #pragma unroll
        for (int k = opcount - 1; k >= 0; k--) {
            operands[k] = pop_stack(s, tid, debug...);
        }

        // Apply operation and push the result to the stack
        push_stack(s, tid, operation(operands), debug...);

        // Save consumed operand as intermediate value for backpropagation
        #pragma unroll
        for (int k = 0; k < opcount; ++k) {
            push_intermediate(s, tid, operands[k], debug...);
        }
    }

    if constexpr (proptype == BACK) {
        // Pop incoming gradient from the stack
        const real incoming_grad = pop_stack(s, tid, debug...);
        vm_debug_print(tid, "  incoming_grad=%f", incoming_grad);

        // Pop intermediate calculation results from forward propagation from intermediate_d
        real intermediates[opcount];
        #pragma unroll
        for (int k = opcount - 1; k >= 0; k--) {
            vm_debug_print(tid, "k=%d: intermediate_index+k=%d\n", k, inst.intermediate_index + k);
            const real intermediate = read_intermediate(s, tid, inst.intermediate_index + k, debug...);
            intermediates[k] = intermediate;
            vm_debug_print(tid, "  intermediates[%d]=%f", k, intermediate);
        }

        // Calculate local gradients
        real local_grad[opcount];
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
            push_stack(s, tid, local_grad[k], debug...);
        }
    }
}

}

#endif
