// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "subtree_mutation.hpp"
#include "mutator/subtree_mutator.hpp"

namespace qsr {
    std::shared_ptr<BaseMutator> SubtreeMutation::get_mutator(int nvars, int nweights) {
        return std::make_shared<SubtreeMutator>(nvars, nweights, max_depth_increment, max_depth, mutation_probability);
    }
}

