// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef RUNNERS_GPU_BASE_HPP
#define RUNNERS_GPU_BASE_HPP

#include "base.hpp"

#include "util/hip.hpp"

namespace qsr {
    class GPUBaseRunner : public BaseRunner {
    protected:
        HIPState hipState;

        dim3 gridDim;

        dim3 blockDim;

        int nthreads;

        GPUBaseRunner(int nweights);

        template <typename K, typename ...T>
        inline void launch_kernel(K kernel, T... args) {
            hipLaunchKernelGGL(kernel, gridDim, blockDim, 0, hipState.stream, args...);
        }

        inline void synchronize() {
            HIP_CALL(hipStreamSynchronize(hipState.stream));
        }

        void calculate_kernel_dims(int nthreads);
    };
}

#endif