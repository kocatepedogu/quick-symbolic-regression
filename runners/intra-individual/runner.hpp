// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef INTRA_RUNNER_HPP
#define INTRA_RUNNER_HPP

#include "../base.hpp"

#include "../../expressions/expression.hpp"
#include "../../dataset/dataset.hpp"

#include </usr/lib/clang/20/include/omp.h>

#include "../../util/hip.hpp"
#include "../../vm/vm_types.hpp"

namespace intra_individual {
    struct VirtualMachineResult {
        std::shared_ptr<Array1D<float>> weights_d;
        std::shared_ptr<Array1D<float>> loss_d;
    };

    class Runner : public BaseRunner {
    private:
        std::shared_ptr<Dataset> dataset;

        const int nweights;

        HIPState hipState;

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

    public:
        Runner(std::shared_ptr<Dataset> dataset, const int nweights);

        void run(c_inst_1d code, int code_length, int epochs, float learning_rate);

        void run(std::vector<Expression>& population, int epochs, float learning_rate) override;
    };
}

#endif