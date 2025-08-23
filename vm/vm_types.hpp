// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef INTRA_VM_TYPES_HPP
#define INTRA_VM_TYPES_HPP

#include "../../../util/arrays/array1d.hpp"
#include "../../../util/arrays/array2d.hpp"

namespace qsr {

typedef float *const __restrict__ real_1d;
typedef const float *const __restrict__ c_real_1d;

enum ParallelismType {
    INTRA_INDIVIDUAL,
    INTER_INDIVIDUAL,
    HYBRID
};

enum PropagationType {
    FORWARD,
    BACK
};

/**
 * @brief Represents the stack and intermediate array state of the virtual machine.
 */
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

/**
 * @brief Represents the dataset of the virtual machine.
 */
struct DataState {
    /**
      * @brief Input feature matrix
     */
    const Ptr2D<float> &X_d;

    /**
      * @brief Target vector
     */
    const Ptr1D<float> &y_d;

    /**
      * @brief All arguments constructor
     */
    constexpr DataState(const Ptr2D<float> &X_d, const Ptr1D<float> &y_d) :
        X_d(X_d), y_d(y_d) {}
};

/**
 * @brief Represents the state of the virtual machine with respect to trainable weights.
 */
template <typename Weights>
struct WeightState {
    /**
      * @brief The weights array.
     */
    Weights &weights_d;

    /**
      * @brief The gradients array.
     */
    Ptr2D<float> &weights_grad_d;

    /**
      * @brief All arguments constructor
     */
    constexpr WeightState(
        Weights &weights_d, Ptr2D<float> &weights_grad_d) :
        weights_d(weights_d), weights_grad_d(weights_grad_d) {}
};

}

#endif