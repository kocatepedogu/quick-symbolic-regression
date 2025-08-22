// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef INTRA_VM_TYPES_HPP
#define INTRA_VM_TYPES_HPP

#include "../compiler/ir.hpp"

#include "../../../util/arrays/array2d.hpp"

namespace qsr {

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
    INTER_INDIVIDUAL,
    HYBRID
};

enum PropagationType {
    FORWARD,
    BACK
};

struct StackState {
    Ptr2D<float> &stack_d;
    Ptr2D<float> &intermediate_d;
    int& stack_pointer;
    int& intermediate_pointer;

    constexpr StackState(
        Ptr2D<float> &stack_d, 
        Ptr2D<float> &intermediate_d,
        int& stack_pointer, 
        int& intermediate_pointer) :
        stack_d(stack_d), 
        intermediate_d(intermediate_d),
        stack_pointer(stack_pointer), 
        intermediate_pointer(intermediate_pointer) {}
};

/**
 * @brief Represents the control state of the virtual machine.
 */
template <typename Code>
struct ControlState {
    /**
      * @brief Global Thread ID of the current thread.
     */
    const int  &tid;

    /**
      * @brief The index of the datapoint currently being processed by the thread.
     */
    const int  &datapoint_idx;

    /**
      * @brief The length of the bytecode array.
     */
    const int  &bytecode_length;

    /**
      * @brief The bytecode array containing the instructions for the virtual machine.
     */
    const Code &bytecode;

    /**
      * @brief The current program counter of the virtual machine.
     */
    int  &program_counter;

    /**
      * @brief All arguments constructor
     */
    constexpr ControlState(
        const int  &tid, const int  &datapoint_idx,
        const int  &bytecode_length, const Code &bytecode, int  &program_counter) :
        tid(tid), datapoint_idx(datapoint_idx),
        bytecode(bytecode), bytecode_length(bytecode_length), program_counter(program_counter) {}
};

}

#endif