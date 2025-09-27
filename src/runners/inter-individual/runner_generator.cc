// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "runners/inter-individual/runner_generator.hpp"
#include "runners/inter-individual/runner.hpp"

namespace qsr {
namespace inter_individual {
    std::shared_ptr<BaseRunner> RunnerGenerator::generate(int nweights, const HIPState *hipState) {
        return std::make_shared<Runner>(nweights, use_cache, hipState);
    }
}
}