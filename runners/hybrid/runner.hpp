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
        std::shared_ptr<Dataset> dataset;

        const int nweights;

        HIPState config;

    public:
        Runner(std::shared_ptr<Dataset> dataset, int nweights);

        void run(const inter_individual::Program &program, int epochs, float learning_rate, Array1D<float> &loss_d, Ptr2D<float> &weights_d);

        void run(std::vector<Expression>& population, int epochs, float learning_rate) override;
    };
}

#endif