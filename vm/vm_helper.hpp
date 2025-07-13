#ifndef VM_HELPER_HPP
#define VM_HELPER_HPP

#include <hip/hip_runtime.h>

constexpr int reduction_threads_per_block = 32;

__global__ void reduce_sum(const float *const input, float *const block_sums, int n);

#endif