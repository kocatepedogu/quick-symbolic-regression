// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "genetic/mutation/point_mutation.hpp"
#include "genetic/mutation/mutator/point_mutator.hpp"

namespace qsr {
    std::shared_ptr<BaseMutator> PointMutation::get_mutator(const Config &config) {
        return std::make_shared<PointMutator>(config, mutation_probability);
    }
}