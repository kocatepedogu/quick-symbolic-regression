// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef VM_GRADIENTS_HPP
#define VM_GRADIENTS_HPP

#include "vm_types.hpp"
#include <hip/hip_runtime.h>

namespace qsr {

__device__ __host__
static inline void grad_add(c_real_1d &i, real_1d &o) {
    o[0] = 1_r;
    o[1] = 1_r;
}

__device__ __host__
static inline void grad_sub(c_real_1d &i, real_1d &o) {
    o[0] = 1_r;
    o[1] = -1_r;
}

__device__ __host__
static inline void grad_mul(c_real_1d &i, real_1d &o) {
    o[0] = i[1];
    o[1] = i[0];
}

__device__ __host__
static inline void grad_div(c_real_1d &i, real_1d &o) {
    o[0] = 1_r/i[1];
    o[1] = -i[0]/(i[1]*i[1]);
}

__device__ __host__
static inline void grad_sin(c_real_1d &i, real_1d &o) {
    o[0] = cos(i[0]);
}

__device__ __host__
static inline void grad_cos(c_real_1d &i, real_1d &o) {
    o[0] = -sin(i[0]);
}

__device__ __host__
static inline void grad_exp(c_real_1d &i, real_1d &o) {
    o[0] = exp(i[0]);
}

__device__ __host__
static inline void grad_relu(c_real_1d &i, real_1d &o) {
    o[0] = i[0] > 0.0_r ? 1_r : 0_r;
}

}

#endif
