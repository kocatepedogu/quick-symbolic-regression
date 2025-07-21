#ifndef VM_FUNCTIONS_HPP
#define VM_FUNCTIONS_HPP

#include <hip/hip_runtime.h>

__device__
static inline float forward_add(const float *const __restrict__ &x) {
    return x[0] + x[1];
}

__device__
static inline float forward_sub(const float *const __restrict__ &x) {
    return x[0] - x[1];
}

__device__
static inline float forward_mul(const float *const __restrict__ &x) {
    return x[0] * x[1];
}

__device__
static inline float forward_div(const float *const __restrict__ &x) {
    return x[0] / x[1];
}

__device__
static inline float forward_sin(const float *const __restrict__ &x) {
    return sin(x[0]);
}

__device__
static inline float forward_cos(const float *const __restrict__ &x) {
    return cos(x[0]);
}

__device__
static inline float forward_exp(const float *const __restrict__ &x) {
    return exp(x[0]);
}

#endif
