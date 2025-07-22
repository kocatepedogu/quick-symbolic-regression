// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef INTRA_VM_HPP
#define INTRA_VM_HPP

#include "../../dataset/dataset.hpp"

#include <hip/hip_runtime.h>
#include </usr/lib/clang/20/include/omp.h>

#include "../../vm/vm_types.hpp"

namespace intra_individual {
    class VirtualMachine {
    public:
        VirtualMachine(const Dataset& dataset, hipStream_t& stream, int nweights, omp_lock_t& print_lock);

        void fit(c_inst_1d code, int code_length, int epochs, float learning_rate);

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

        real_2d_mut stack_d;
        real_2d_mut intermediate_d;
        real_1d_mut weights_d;
        real_2d_mut weights_grad_d;
        real_2d_mut weights_grad_reduced_sum_d;

        omp_lock_t &print_lock;
    };
}

#endif