#include <hip/hip_runtime.h>

#include "vm_debug.hpp"

enum PropagationType {
    FORWARD,
    BACK
};

struct StackState {
    float **&stack_d;
    float **&stack_intermediate_d;
    int& stack_pointer;
    int& stack_intermediate_pointer;

    constexpr StackState(
        float **&stack_d, 
        float **&stack_intermediate_d,
        int& stack_pointer, 
        int& stack_intermediate_pointer) :
        stack_d(stack_d), 
        stack_intermediate_d(stack_intermediate_d),
        stack_pointer(stack_pointer), 
        stack_intermediate_pointer(stack_intermediate_pointer) {}
};


template <PropagationType proptype> __device__
static inline void propagate_immediate(int tid, const float& value, const StackState& s) {
    if constexpr (proptype == FORWARD) {
        const float operand1 = value;
        vm_debug_print("push %f", operand1);
        s.stack_d[s.stack_pointer++][tid] = operand1;
    }
}


template <PropagationType proptype, typename F, typename G> __device__ 
static inline void propagate_binary(int tid, F operation, G gradient, const StackState& s) {
    if constexpr (proptype == FORWARD) {
        // Pop operands from the stack
        const float operand1 = s.stack_d[--s.stack_pointer][tid];
        const float operand2 = s.stack_d[--s.stack_pointer][tid];

        // Apply operation and push the result to the stack
        s.stack_d[s.stack_pointer++][tid] = operation(operand2, operand1);

        // Save consumed operands as intermediate values for backpropagation
        s.stack_intermediate_d[s.stack_intermediate_pointer++][tid] = operand2;
        s.stack_intermediate_d[s.stack_intermediate_pointer++][tid] = operand1;
    }
}


template <PropagationType proptype, typename F, typename G> __device__ 
static inline void propagate_unary(int tid, F operation, G gradient, const StackState& s) {
    if constexpr (proptype == FORWARD) {
        // Pop operand from the stack
        const float operand1 = s.stack_d[--s.stack_pointer][tid];

        // Apply operation and push the result to the stack
        s.stack_d[s.stack_pointer++][tid] = operation(operand1);

        // Save consumed operand as intermediate value for backpropagation
        s.stack_intermediate_d[s.stack_intermediate_pointer++][tid] = operand1;
    }
}
