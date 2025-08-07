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

namespace qsr {
namespace intra_individual {
    class Runner : public BaseRunner {
    private:
        std::shared_ptr<Dataset> dataset;

        const int nweights;

        HIPState hipState;

        dim3 gridDim;
        dim3 blockDim;
        dim3 reduction_grid_dim;
        dim3 reduction_block_dim;

        Array1D<float> loss_d;
        Array2D<float> stack_d;
        Array2D<float> intermediate_d;
        Array1D<float> weights_d;
        Array2D<float> weights_grad_d;

    public:
        Runner(std::shared_ptr<Dataset> dataset, const int nweights);

        void run(c_inst_1d code, int code_length, int stack_length, int intermediate_length, int epochs, float learning_rate);

        void run(std::vector<Expression>& population, int epochs, float learning_rate) override;
    };
}
}

#endif