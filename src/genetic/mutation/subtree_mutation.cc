// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "genetic/mutation/subtree_mutation.hpp"
#include "genetic/mutation/mutator/subtree_mutator.hpp"

namespace qsr {
    std::shared_ptr<BaseMutator> SubtreeMutation::get_mutator(const Config &config) {
        return std::make_shared<SubtreeMutator>(config, mutation_probability, max_depth_increment);
    }
}

