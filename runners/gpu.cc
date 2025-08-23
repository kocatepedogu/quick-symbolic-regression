// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "gpu.hpp"
#include "base.hpp"

using namespace qsr;

GPUBaseRunner::GPUBaseRunner(int nweights) : 
    BaseRunner(nweights) {}

void GPUBaseRunner::calculate_kernel_dims(int nthreads) {
    this->nthreads = nthreads;

    // Use the maximum possible number of threads per block, unless fewer is required in total
    const int threads_per_block = std::min(nthreads, hipState.props.maxThreadsPerBlock);
    blockDim = dim3(threads_per_block);

    // Calculate number of blocks needed
    const int nblocks = (nthreads + threads_per_block - 1) / threads_per_block;
    gridDim = dim3(nblocks);
}