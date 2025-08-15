// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "default.hpp"
#include "mutator/default.hpp"

namespace qsr {
    std::shared_ptr<BaseMutator> DefaultMutation::get_mutator(int nvars, int nweights) {
        return std::make_shared<DefaultMutator>(nvars, nweights, max_depth_increment, max_depth, mutation_probability);
    }
}

