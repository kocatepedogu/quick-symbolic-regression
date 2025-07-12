#ifndef VM_DEBUG_HPP
#define VM_DEBUG_HPP

#include <hip/hip_runtime.h>

/// Whether to print each executed instruction or not
constexpr bool vm_debug_messages = true;

/// Which thread is going to print debug messages
constexpr int vm_debug_tid = 2;

template <typename... T> __device__ 
static inline void vm_debug_print(T ...args) {
    if constexpr (vm_debug_messages) {
        if (blockDim.x * blockIdx.x + threadIdx.x == vm_debug_tid) {
            printf(args...);
            printf("\n");
        }
    }
}

#endif
