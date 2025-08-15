// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef MUTATION_SUBTREE_HPP
#define MUTATION_SUBTREE_HPP

#include "base.hpp"

#include <memory>

namespace qsr {

class SubtreeMutation : public BaseMutation {
public:
    constexpr SubtreeMutation(int max_depth_increment, float mutation_probability) :
        max_depth_increment(max_depth_increment),
        mutation_probability(mutation_probability) {}

    virtual std::shared_ptr<BaseMutator> get_mutator(int nvars, int nweights, int max_depth) override;

private:
    int max_depth_increment;

    float mutation_probability;
};

}

#endif