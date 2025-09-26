// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef INTRAINDIVIDUAL_RUNNER_GENERATOR_HPP
#define INTRAINDIVIDUAL_RUNNER_GENERATOR_HPP

#include "runners/runner_generator_base.hpp"

namespace qsr {
namespace intra_individual {
    class RunnerGenerator : public BaseRunnerGenerator {
    private:
        bool use_cache;
    public:
        explicit RunnerGenerator(const bool use_cache) : use_cache(use_cache) {}

        std::shared_ptr<BaseRunner> generate(int nweights, const HIPState *hipState) override;
    };
}}

#endif