// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef VM_HPP
#define VM_HPP

#include "../compiler/bytecode.hpp"
#include "../dataset/dataset.hpp"

#include <hip/hip_runtime.h>

class VirtualMachine {
public:
    VirtualMachine(const Dataset& dataset, int nweigths);

    void fit(const Program& program);

    ~VirtualMachine();

private:
    const Dataset& dataset;
    const int nweights;

    int deviceId;
    hipDeviceProp_t props;

    dim3 gridDim;
    dim3 blockDim;
    dim3 reduction_grid_dim;
    dim3 reduction_block_dim;

    float **stack_d;
    float **intermediate_d;
    float *weights_d;
    float **weights_grad_d;
    float **block_sums;
};

#endif