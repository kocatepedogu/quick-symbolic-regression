#include "vm_helper.hpp"

__global__ void reduce_sum(const float *const input, float *const block_sums, int n) {
    // Shared memory for block-level reduction
    __shared__ float sharedMem[reduction_threads_per_block];
    
    // Thread and block indices
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sharedMem[tid] = (globalIdx < n) ? input[globalIdx] : 0.0f;
    __syncthreads();
    
    // Parallel reduction within the block
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedMem[tid] += sharedMem[tid + stride];
        }
        __syncthreads();
    }
    
    // First thread of each block writes block sum to output
    if (tid == 0) {
        block_sums[bid] = sharedMem[0];
    }
}