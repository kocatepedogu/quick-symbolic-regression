// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef INTRAINDIVIDUAL_RUNNER_GENERATOR_HPP
#define INTRAINDIVIDUAL_RUNNER_GENERATOR_HPP

#include "../runner_generator_base.hpp"

namespace qsr {
namespace intra_individual {
    class RunnerGenerator : public BaseRunnerGenerator {
    public:
        std::shared_ptr<BaseRunner> generate(std::shared_ptr<const Dataset> dataset, int nweights) override;
    };
}}

#endif