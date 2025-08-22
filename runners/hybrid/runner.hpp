// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef HYBRID_RUNNER_HPP
#define HYBRID_RUNNER_HPP

#include "../base.hpp"

#include "../inter-individual/program/program.hpp"

#include "../../expressions/expression.hpp"
#include "../../dataset/dataset.hpp"

#include <memory>

namespace qsr::hybrid {
    class Runner : public BaseRunner {
    private:
        const int nweights;

        Array2D<float> weights_d;

        Array1D<float> loss_d;

        Array2D<float> stack_d;

        Array2D<float> intermediate_d;

        Array2D<float> weights_grad_d;

        HIPState config;

        dim3 gridDim;

        dim3 blockDim;

        int nthreads;

        void calculate_kernel_dims(const inter_individual::Program &program);

        void initialize_weights_and_losses(std::vector<Expression>& population);

        void save_weights_and_losses(std::vector<Expression>& population);

    public:
        Runner(int nweights);

        void run(std::vector<Expression>& population, std::shared_ptr<const Dataset> dataset, int nepochs, float learning_rate) override;
    };
}

#endif