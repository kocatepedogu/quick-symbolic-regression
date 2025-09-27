// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef QUICKSR_HALF_H
#define QUICKSR_HALF_H

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

/* Unary functions of type __half */

#ifndef __HIP_DEVICE_COMPILE__

__host__
constexpr __half sin(__half x) {
    return __float2half(sinf(__half2float(x)));
}

__host__
constexpr __half cos(__half x) {
    return __float2half(cosf(__half2float(x)));
}

__host__
constexpr __half exp(__half x) {
    return __float2half(expf(__half2float(x)));
}

__host__
constexpr __half ceil(__half x) {
    return static_cast<__half>(static_cast<int>(x));
}

#else

__device__
constexpr __half sin(__half x) {
    return hsin(x);
}

__device__
constexpr __half cos(__half x) {
    return hcos(x);
}

__device__
constexpr __half exp(__half x) {
    return hexp(x);
}

__device__
constexpr __half ceil(__half x) {
    return hceil(x);
}

#endif

/**
 * @brief Performs atomic addition on a 16-bit __half value in global or shared memory.
 *
 * @param address The memory address of the __half value to be modified.
 * @param val The __half value to add to the value at the specified address.
 */
__device__ __forceinline__
void atomicAdd(__half *address, __half val) {
    // The address of the 32-bit integer that contains the target __half value.
    // This relies on the address being aligned to at least 2 bytes.
    auto* base_address = reinterpret_cast<unsigned int*>(
        reinterpret_cast<uintptr_t>(address) & ~3
    );

    // Determine if the target __half is in the lower or upper 16 bits of the 32-bit word.
    const bool is_upper_half = (reinterpret_cast<uintptr_t>(address) & 2)!= 0;

    unsigned int old_val, new_val;

    do {
        // Atomically read the entire 32-bit word.
        old_val = *base_address;

        // Safely extract the 16 bits we are interested in using bitwise operations.
        const unsigned short old_half_bits = is_upper_half?
                                       (old_val >> 16) :
                                       (old_val & 0xFFFF);

        // Use the HIP intrinsic to safely reinterpret the bits as a __half.
        __half old_half = __ushort_as_half(old_half_bits);

        // Perform the floating-point addition.
        const __half new_half = old_half + val;

        // Use the HIP intrinsic to safely reinterpret the result back to bits.
        const unsigned short new_half_bits = __half_as_ushort(new_half);

        // Prepare the new 32-bit word by replacing the relevant 16 bits.
        if (is_upper_half) {
            new_val = (old_val & 0x0000FFFF) | (static_cast<unsigned int>(new_half_bits) << 16);
        } else {
            new_val = (old_val & 0xFFFF0000) | new_half_bits;
        }

        // Attempt the atomic compare-and-swap. The loop continues if another thread
        // changed `old_val` between our read and our atomicCAS attempt.
    } while (atomicCAS(base_address, old_val, new_val)!= old_val);
}

#endif //QUICKSR_HALF_H