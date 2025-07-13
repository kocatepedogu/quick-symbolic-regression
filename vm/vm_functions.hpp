#include <hip/hip_runtime.h>

__device__
static inline float forward_add(const float *const& x) {
    return x[0] + x[1];
}

__device__
static inline float forward_sub(const float *const& x) {
    return x[0] - x[1];
}

__device__
static inline float forward_mul(const float *const& x) {
    return x[0] * x[1];
}

__device__
static inline float forward_div(const float *const& x) {
    return x[0] / x[1];
}

__device__
static inline float forward_sin(const float *const& x) {
    return sin(x[0]);
}

__device__
static inline float forward_cos(const float *const& x) {
    return cos(x[0]);
}

__device__
static inline float forward_exp(const float *const& x) {
    return exp(x[0]);
}
