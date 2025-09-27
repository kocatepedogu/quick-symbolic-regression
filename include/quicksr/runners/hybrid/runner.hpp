// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef HYBRID_RUNNER_HPP
#define HYBRID_RUNNER_HPP

#include "runners/gpu.hpp"
#include "runners/inter-individual/cache/cache.hpp"

#include "expressions/expression.hpp"
#include "dataset/dataset.hpp"

#include <memory>

namespace qsr::hybrid {
    class Runner : public GPUBaseRunner {
    private:
        Array2D<real> weights_d;

        inter_individual::Cache population_cache;

        bool use_cache;

        void initialize_weights_and_losses(std::vector<Expression>& population);

        void save_weights_and_losses(std::vector<Expression>& population);

    public:
        Runner(int nweights, bool use_cache, const HIPState *hipState);

        void run(std::vector<Expression>& population, std::shared_ptr<const Dataset> dataset, int nepochs, double learning_rate) override;
    };
}

#endif