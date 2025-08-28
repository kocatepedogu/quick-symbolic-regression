// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef INTER_RUNNER_HPP
#define INTER_RUNNER_HPP

#include "runners/gpu.hpp"
#include "expressions/expression.hpp"
#include "dataset/dataset.hpp"
#include "util/arrays/array2d.hpp"

#include </usr/lib/clang/20/include/omp.h>

namespace qsr {
namespace inter_individual {
    class Runner : public GPUBaseRunner {
    private:
        Array2D<float> weights_d;

        void initialize_weights_and_losses(std::vector<Expression>& population);

        void save_weights_and_losses(std::vector<Expression>& population);

    public:
        Runner(int nweights);

        void run(std::vector<Expression>& population, std::shared_ptr<const Dataset> dataset, int epochs, float learning_rate) override;
    };
}
}

#endif