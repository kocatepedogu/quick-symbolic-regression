// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef INTER_VM_HPP
#define INTER_VM_HPP

#include "../../../util/hip.hpp"
#include "../../../vm/vm_types.hpp"
#include "../../../dataset/dataset.hpp"
#include "../program/program.hpp"

#include <hip/hip_runtime.h>

namespace inter_individual {
    struct VirtualMachineResult {
        std::unique_ptr<Array2D<float>> weights_d;
        std::unique_ptr<Array1D<float>> loss_d;
    };

    class VirtualMachine {
    public:
        VirtualMachine(std::shared_ptr<Dataset> dataset, int nweights, hipStream_t &stream);

        VirtualMachineResult fit(const Program& program, int epochs, float learning_rate);

    private:
        std::shared_ptr<Dataset> dataset;
        const int nweights;

        int device_id;
        hipDeviceProp_t props;

        hipStream_t& stream;
    };
};

#endif