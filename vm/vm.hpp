// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef VM_HPP
#define VM_HPP

#include "../compiler/bytecode.hpp"
#include "../dataset/dataset.hpp"

#include <hip/hip_runtime.h>
#include </usr/lib/clang/20/include/omp.h>

class VirtualMachine {
public:
    VirtualMachine(const Dataset& dataset, hipStream_t& stream, int nweights, omp_lock_t& print_lock);

    void fit(const Program& program, int epochs = 500, float learning_rate = 5e-4);

    ~VirtualMachine();

private:
    const Dataset& dataset;
    const int nweights;

    int device_id;

    hipDeviceProp_t props;
    hipStream_t& stream;

    dim3 gridDim;
    dim3 blockDim;
    dim3 reduction_grid_dim;
    dim3 reduction_block_dim;

    float **stack_d;
    float **intermediate_d;
    float  *weights_d;
    float **weights_grad_d;
    float **weights_grad_reduced_sum_d;

    omp_lock_t &print_lock;
};

#endif