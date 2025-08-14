// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef MUTATION_DEFAULT_HPP
#define MUTATION_DEFAULT_HPP

#include "base.hpp"
#include "mutator/default.hpp"

#include <memory>

namespace qsr {

class DefaultMutation : public BaseMutation {
public:
    constexpr DefaultMutation(int max_depth, float mutation_probability) :
        max_depth(max_depth),
        mutation_probability(mutation_probability) {}

    virtual std::shared_ptr<BaseMutator> get_mutator(int nvars, int nweights) override;

private:
    int max_depth;
    float mutation_probability;
};

}

#endif