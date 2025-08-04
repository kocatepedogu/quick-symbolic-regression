// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef VM_FUNCTIONS_HPP
#define VM_FUNCTIONS_HPP

#include "vm_types.hpp"
#include <hip/hip_runtime.h>

namespace qsr {

__device__
static inline float forward_add(c_real_1d &x) {
    return x[0] + x[1];
}

__device__
static inline float forward_sub(c_real_1d &x) {
    return x[0] - x[1];
}

__device__
static inline float forward_mul(c_real_1d &x) {
    return x[0] * x[1];
}

__device__
static inline float forward_div(c_real_1d &x) {
    return x[0] / x[1];
}

__device__
static inline float forward_sin(c_real_1d &x) {
    return sin(x[0]);
}

__device__
static inline float forward_cos(c_real_1d &x) {
    return cos(x[0]);
}

__device__
static inline float forward_exp(c_real_1d &x) {
    return exp(x[0]);
}

}

#endif
