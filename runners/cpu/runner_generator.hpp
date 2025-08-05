// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef CPU_RUNNER_GENERATOR_HPP
#define CPU_RUNNER_GENERATOR_HPP

#include "../runner_generator_base.hpp"

namespace qsr::cpu {
    class RunnerGenerator : public BaseRunnerGenerator {
    public:
        std::shared_ptr<BaseRunner> generate(std::shared_ptr<Dataset> dataset, int nweights);
    };
}

#endif