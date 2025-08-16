// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "hoist_mutation.hpp"
#include "mutator/hoist_mutator.hpp"

namespace qsr {
    std::shared_ptr<BaseMutator> HoistMutation::get_mutator(const Config &config) {
        return std::make_shared<HoistMutator>(mutation_probability);
    }
}

