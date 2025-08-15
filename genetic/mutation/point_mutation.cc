// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "point_mutation.hpp"
#include "mutator/point_mutator.hpp"

namespace qsr {
    std::shared_ptr<BaseMutator> PointMutation::get_mutator(int nvars, int nweights, int max_depth) {
        return std::make_shared<PointMutator>(nvars, nweights, mutation_probability);
    }
}