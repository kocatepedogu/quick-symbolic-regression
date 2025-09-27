// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef VM_FUNCTIONS_HPP
#define VM_FUNCTIONS_HPP

#include "vm_types.hpp"

namespace qsr {

__device__ __host__
static inline real forward_add(c_real_1d &x) {
    return x[0] + x[1];
}

__device__ __host__
static inline real forward_sub(c_real_1d &x) {
    return x[0] - x[1];
}

__device__ __host__
static inline real forward_mul(c_real_1d &x) {
    return x[0] * x[1];
}

__device__ __host__
static inline real forward_div(c_real_1d &x) {
    return x[0] / x[1];
}

__device__ __host__
static inline real forward_sin(c_real_1d &x) {
    return sin(x[0]);
}

__device__ __host__
static inline real forward_cos(c_real_1d &x) {
    return cos(x[0]);
}

__device__ __host__
static inline real forward_exp(c_real_1d &x) {
    return exp(x[0]);
}

__device__ __host__
static inline real forward_relu(c_real_1d &x) {
    return x[0] > static_cast<real>(0) ? x[0] : static_cast<real>(0);
}

}

#endif
