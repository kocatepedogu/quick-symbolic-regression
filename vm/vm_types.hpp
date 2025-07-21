#ifndef INTRA_VM_TYPES_HPP
#define INTRA_VM_TYPES_HPP

enum ParallelismType {
    INTRA_INDIVIDUAL,
    INTER_INDIVIDUAL
};

enum PropagationType {
    FORWARD,
    BACK
};

struct StackState {
    float *const __restrict__ *const __restrict__ &stack_d;
    float *const __restrict__ *const __restrict__ &intermediate_d;
    int& stack_pointer;
    int& intermediate_pointer;

    constexpr StackState(
        float *const __restrict__ *const __restrict__ &stack_d, 
        float *const __restrict__ *const __restrict__ &intermediate_d,
        int& stack_pointer, 
        int& intermediate_pointer) :
        stack_d(stack_d), 
        intermediate_d(intermediate_d),
        stack_pointer(stack_pointer), 
        intermediate_pointer(intermediate_pointer) {}
};

#endif