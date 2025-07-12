#include <hip/hip_runtime.h>

__device__
static inline float forward_add(float a, float b) {
    return a + b;
}

__device__
static inline float forward_sub(float a, float b) {
    return a - b;
}

__device__
static inline float forward_mul(float a, float b) {
    return a * b;
}

__device__
static inline float forward_div(float a, float b) {
    return a / b;
}

__device__
static inline float forward_sin(float a) {
    return sin(a);
}

__device__
static inline float forward_cos(float a) {
    return cos(a);
}

__device__
static inline float forward_exp(float a) {
    return exp(a);
}
