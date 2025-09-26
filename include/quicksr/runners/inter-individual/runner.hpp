// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef INTER_RUNNER_HPP
#define INTER_RUNNER_HPP

#include "runners/inter-individual/cache/cache.hpp"
#include "runners/gpu.hpp"
#include "expressions/expression.hpp"
#include "dataset/dataset.hpp"
#include "util/arrays/array2d.hpp"

#include <omp.h>

namespace qsr {
namespace inter_individual {
    class Runner : public GPUBaseRunner {
    private:
        Array2D<real> weights_d;

        Cache population_cache;

        bool use_cache;

        void initialize_weights_and_losses(std::vector<Expression>& population);

        void save_weights_and_losses(std::vector<Expression>& population);

    public:
        Runner(int nweights, bool use_cache);

        void run(std::vector<Expression>& population, std::shared_ptr<const Dataset> dataset, int epochs, double learning_rate) override;
    };
}
}

#endif
