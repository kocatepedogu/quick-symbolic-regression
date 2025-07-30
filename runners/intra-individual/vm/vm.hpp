// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef INTRA_VM_HPP
#define INTRA_VM_HPP

#include "../../../dataset/dataset.hpp"

#include <hip/hip_runtime.h>
#include </usr/lib/clang/20/include/omp.h>

#include "../../../vm/vm_types.hpp"
#include "../../../util/hip.hpp"

namespace intra_individual {
    struct VirtualMachineResult {
        std::shared_ptr<Array1D<float>> weights_d;
        std::shared_ptr<Array1D<float>> loss_d;
    };

    class VirtualMachine {
    public:
        VirtualMachine(std::shared_ptr<Dataset> dataset, hipStream_t& stream, int nweights, omp_lock_t& print_lock);

        VirtualMachineResult fit(c_inst_1d code, int code_length, int epochs, float learning_rate);

    private:
        std::shared_ptr<Dataset> dataset;
        const int nweights;

        int device_id;

        hipDeviceProp_t props;
        hipStream_t& stream;

        dim3 gridDim;
        dim3 blockDim;
        dim3 reduction_grid_dim;
        dim3 reduction_block_dim;

        std::shared_ptr<Array1D<float>> loss_d;
        std::shared_ptr<Array2D<float>> stack_d;
        std::shared_ptr<Array2D<float>> intermediate_d;
        std::shared_ptr<Array1D<float>> weights_d;
        std::shared_ptr<Array2D<float>> weights_grad_d;
        std::shared_ptr<Array2D<float>> weights_grad_reduced_sum_d;

        omp_lock_t &print_lock;
    };
}

#endif