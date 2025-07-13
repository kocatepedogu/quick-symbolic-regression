#include <hip/hip_runtime.h>

__device__
static inline void grad_add(const float& i1, const float& i2, float& o1, float& o2) {
    o1 = 1;
    o2 = 1;
}

__device__
static inline void grad_sub(const float& i1, const float& i2, float& o1, float& o2) {
    o1 = 1;
    o2 = -1;
}

__device__
static inline void grad_mul(const float& i1, const float& i2, float& o1, float& o2) {
    o1 = i2;
    o2 = i1;
}

__device__
static inline void grad_div(const float& i1, const float& i2, float& o1, float& o2) {
    o1 = 1/i2;
    o2 = -i1/(i2*i2);
}

__device__
static inline void grad_sin(const float& i1, float& o1) {
    o1 = cos(i1);
}

__device__
static inline void grad_cos(const float& i1, float& o1) {
    o1 = -sin(i1);
}

__device__
static inline void grad_exp(const float& i1, float& o1) {
    o1 = exp(i1);
}
