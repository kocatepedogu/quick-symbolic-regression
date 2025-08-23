// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef HYBRID_RUNNER_HPP
#define HYBRID_RUNNER_HPP

#include "../gpu.hpp"

#include "../../expressions/expression.hpp"
#include "../../dataset/dataset.hpp"

#include <memory>

namespace qsr::hybrid {
    class Runner : public GPUBaseRunner {
    private:
        Array2D<float> weights_d;

        void initialize_weights_and_losses(std::vector<Expression>& population);

        void save_weights_and_losses(std::vector<Expression>& population);

    public:
        Runner(int nweights);

        void run(std::vector<Expression>& population, std::shared_ptr<const Dataset> dataset, int nepochs, float learning_rate) override;
    };
}

#endif