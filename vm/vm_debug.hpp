#ifndef INTRA_VM_DEBUG_HPP
#define INTRA_VM_DEBUG_HPP

#include <hip/hip_runtime.h>
#include "vm_types.hpp"

/// Whether to print debug messages or not
constexpr bool vm_debug_messages = false;

/// Which thread is going to print debug messages
constexpr int vm_debug_tid = 2;

template <typename... T> __device__ 
static inline void vm_debug_print(const int& tid, T ...args) {
    if constexpr (vm_debug_messages) {
        if (tid == vm_debug_tid) {
            printf(args...);
            printf("\n");
        }
    }
}

__device__
static inline void vm_debug_print_stack(const int& tid, const StackState& s) {
    if constexpr (vm_debug_messages) {
        if (tid == vm_debug_tid) {
            /* Print stack contents */

            printf("  stack: ");
            for (int i = 0; i < s.stack_pointer; ++i) {
                printf("%f ", s.stack_d[i][tid]);
            }
            printf("\n");

            /* Print intermediate calculation results */

            printf("  intermediate: ");
            for (int i = 0; i < s.intermediate_pointer; ++i) {
                printf("%f ", s.intermediate_d[i][tid]);
            }
            printf("\n");
        }
    }
}

#endif
