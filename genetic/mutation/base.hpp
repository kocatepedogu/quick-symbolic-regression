// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef MUTATION_BASE_HPP
#define MUTATION_BASE_HPP

#include "mutator/base.hpp"
#include "../common/function_set.hpp"

#include <memory>

namespace qsr {

class BaseMutation {
public:
    virtual std::shared_ptr<BaseMutator> get_mutator(int nvars, int nweights, int max_depth, std::shared_ptr<FunctionSet> function_set);
};

}

#endif