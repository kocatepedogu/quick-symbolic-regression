// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef MUTATION_HOIST_HPP
#define MUTATION_HOIST_HPP

#include "base.hpp"

#include <memory>

namespace qsr {

class HoistMutation : public BaseMutation {
public:
    constexpr HoistMutation(float mutation_probability) :
        mutation_probability(mutation_probability) {}

    virtual std::shared_ptr<BaseMutator> get_mutator(const Config &config) override;

private:
    float mutation_probability;
};

}

#endif