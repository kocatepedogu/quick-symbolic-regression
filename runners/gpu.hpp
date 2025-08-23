// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef RUNNERS_GPU_BASE_HPP
#define RUNNERS_GPU_BASE_HPP

#include "base.hpp"

namespace qsr {

class GPUBaseRunner : public BaseRunner {
protected:
    HIPState hipState;

    dim3 gridDim;

    dim3 blockDim;

    GPUBaseRunner(int nweights);
};

}

#endif