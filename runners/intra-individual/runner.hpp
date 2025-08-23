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
        HIPState hipState;

        dim3 gridDim;
        dim3 blockDim;
        dim3 reduction_grid_dim;
        dim3 reduction_block_dim;
        
        Array1D<float> weights_d;
        
        void calculate_kernel_dims(int m);

        void initialize_weights_and_losses(Expression &expression);

        void save_weights_and_losses(Expression &expression);

    public:
        Runner(const int nweights);

        void run(Ptr1D<Instruction> bytecode, int code_length, int stack_length, int intermediate_length, std::shared_ptr<const Dataset> dataset, int epochs, float learning_rate);

        void run(std::vector<Expression>& population, std::shared_ptr<const Dataset> dataset, int epochs, float learning_rate) override;
    };
}
}

#endif