// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef HYBRID_RUNNER_GENERATOR_HPP
#define HYBRID_RUNNER_GENERATOR_HPP

#include "runners/runner_generator_base.hpp"

namespace qsr::hybrid {
    class RunnerGenerator : public BaseRunnerGenerator {
    public:
        std::shared_ptr<BaseRunner> generate(int nweights) override;
    };
}

#endif 