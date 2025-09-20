// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "runners/hybrid/runner_generator.hpp"
#include "runners/hybrid/runner.hpp"

namespace qsr::hybrid {
    std::shared_ptr<BaseRunner> RunnerGenerator::generate(int nweights) {
        return std::make_shared<Runner>(nweights, use_cache);
    }
}