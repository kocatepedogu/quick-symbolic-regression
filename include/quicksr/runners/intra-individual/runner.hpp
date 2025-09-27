// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef INTRA_RUNNER_HPP
#define INTRA_RUNNER_HPP

#include "runners/gpu.hpp"

#include "expressions/expression.hpp"
#include "dataset/dataset.hpp"
#include "runners/intra-individual/cache/cache.hpp"
#include "runners/intra-individual/program/program.hpp"

#include "util/hip.hpp"


namespace qsr {
namespace intra_individual {
    class Runner : public GPUBaseRunner {
    private:
        dim3 reduction_grid_dim;
        
        dim3 reduction_block_dim;
        
        Array1D<real> weights_d;

        Cache population_cache;

        bool use_cache;

        void run(const ProgramIndividual &p, std::shared_ptr<const Dataset> dataset, int epochs, real learning_rate);
        
        void calculate_kernel_dims(int m);

        void initialize_weights_and_losses(Expression &expression);

        void save_weights_and_losses(Expression &expression);

    public:
        Runner(const int nweights, bool use_cache, const HIPState *hipState);

        void run(std::vector<Expression>& population, std::shared_ptr<const Dataset> dataset, int epochs, double learning_rate) override;
    };
}
}

#endif
