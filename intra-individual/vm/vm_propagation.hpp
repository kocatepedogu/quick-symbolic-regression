#ifndef VM_PROPAGATION_HPP
#define VM_PROPAGATION_HPP

#include <hip/hip_runtime.h>
#include "vm_debug.hpp"
#include "vm_types.hpp"

#include "../../compiler/program.hpp"

namespace intra_individual {
    template <PropagationType proptype> __device__
    static inline void propagate_immediate(int tid, const float& immediate, const StackState& s) {
        if constexpr (proptype == FORWARD) {
            s.stack_d[s.stack_pointer++][tid] = immediate;
        }

        if constexpr (proptype == BACK) {
            // Pop gradient from stack, discard the value.
            --s.stack_pointer;
        }
    }

    template <PropagationType proptype> __device__
    static inline void propagate_variable(int tid, const int& variable_index, const StackState& s, 
                                        const float *const __restrict__ *const __restrict__ X_d) {
        if constexpr (proptype == FORWARD) {
            s.stack_d[s.stack_pointer++][tid] = X_d[variable_index][tid];
        }

        if constexpr (proptype == BACK) {
            // Pop gradient from stack, discard the value.
            --s.stack_pointer;
        }
    }

    template <PropagationType proptype> __device__
    static inline void propagate_parameter(int tid, const int& param_index, const StackState& s,
                                        float *const __restrict__ weights, 
                                        float *const __restrict__ *const __restrict__ weights_grad_d) {
        if constexpr (proptype == FORWARD) {
            s.stack_d[s.stack_pointer++][tid] = weights[param_index];
        }

        if constexpr (proptype == BACK) {
            // Pop gradient from stack
            const float incoming_grad = s.stack_d[--s.stack_pointer][tid];
            vm_debug_print(tid, "  incoming_grad=%f", incoming_grad);

            // Add gradient to the total gradient of the associated trainable parameter
            weights_grad_d[param_index][tid] += incoming_grad;
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
}

#endif
