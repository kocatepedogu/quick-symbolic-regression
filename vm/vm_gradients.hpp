#include <hip/hip_runtime.h>

__device__
static inline void grad_add(const float *const& i, float *const& o) {
    o[0] = 1;
    o[1] = 1;
}

__device__
static inline void grad_sub(const float *const& i, float *const& o) {
    o[0] = 1;
    o[1] = -1;
}

__device__
static inline void grad_mul(const float *const& i, float *const& o) {
    o[0] = i[1];
    o[1] = i[0];
}

__device__
static inline void grad_div(const float *const& i, float *const& o) {
    o[0] = 1/i[1];
    o[1] = -i[0]/(i[1]*i[1]);
}

__device__
static inline void grad_sin(const float *const& i, float *const& o) {
    o[0] = cos(i[0]);
}

__device__
static inline void grad_cos(const float *const& i, float *const& o) {
    o[0] = -sin(i[0]);
}

__device__
static inline void grad_exp(const float *const& i, float *const& o) {
    o[0] = exp(i[0]);
}
