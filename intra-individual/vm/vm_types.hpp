#ifndef VM_TYPES_HPP
#define VM_TYPES_HPP

namespace intra_individual {
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
}

#endif