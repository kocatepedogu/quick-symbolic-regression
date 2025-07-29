// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef INTER_VM_HPP
#define INTER_VM_HPP

#include "../../../vm/vm_types.hpp"
#include "../../../dataset/dataset.hpp"
#include "../program/program.hpp"

#include <hip/hip_runtime.h>

namespace inter_individual {
    class VirtualMachine {
    public:
        VirtualMachine(const Dataset& dataset, int nweights, hipStream_t &stream);

        void fit(const Program& program, real_1d loss_d, int epochs, float learning_rate);

    private:
        const Dataset& dataset;
        const int nweights;

        int device_id;
        hipDeviceProp_t props;

        hipStream_t& stream;
    };
};

#endif