#ifndef INTRA_VM_TYPES_HPP
#define INTRA_VM_TYPES_HPP

#include "../compiler/ir.hpp"

typedef Instruction *inst_1d_mut;
typedef Instruction *const __restrict__ inst_1d;
typedef const Instruction *const __restrict__ c_inst_1d;

typedef Instruction **inst_2d_mut;
typedef Instruction *const __restrict__ *const __restrict inst_2d;
typedef const Instruction *const __restrict__ *const __restrict c_inst_2d;

typedef float *real_1d_mut;
typedef float *const __restrict__ real_1d;
typedef const float *const __restrict__ c_real_1d;

typedef float **real_2d_mut;
typedef float *const __restrict__ *const __restrict__ real_2d;
typedef const float *const __restrict__ *const __restrict__ c_real_2d;

enum ParallelismType {
    INTRA_INDIVIDUAL,
    INTER_INDIVIDUAL
};

enum PropagationType {
    FORWARD,
    BACK
};

struct StackState {
    real_2d &stack_d;
    real_2d &intermediate_d;
    int& stack_pointer;
    int& intermediate_pointer;

    constexpr StackState(
        real_2d &stack_d, 
        real_2d &intermediate_d,
        int& stack_pointer, 
        int& intermediate_pointer) :
        stack_d(stack_d), 
        intermediate_d(intermediate_d),
        stack_pointer(stack_pointer), 
        intermediate_pointer(intermediate_pointer) {}
};

#endif