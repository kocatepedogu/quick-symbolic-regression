// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef INTER_RUNNER_HPP
#define INTER_RUNNER_HPP

#include "../base.hpp"

#include "./program/program.hpp"

#include "../../expressions/expression.hpp"
#include "../../dataset/dataset.hpp"

#include "../../../util/hip.hpp"

#include </usr/lib/clang/20/include/omp.h>

namespace inter_individual {
    class Runner : public BaseRunner {
    private:
        std::shared_ptr<Dataset> dataset;

        const int nweights;

        HIPState config;

    public:
        Runner(std::shared_ptr<Dataset> dataset, int nweights);

        void run(const Program &program, int epochs, float learning_rate, Array1D<float> &loss_d, Array2D<float> &weights_d);

        void run(std::vector<Expression>& population, int epochs, float learning_rate) override;
    };
}

#endif