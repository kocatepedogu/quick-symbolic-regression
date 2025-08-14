// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef MUTATOR_BASE_HPP
#define MUTATOR_BASE_HPP

#include "../../../expressions/expression.hpp"

namespace qsr {

class BaseMutator {
public:
    virtual Expression mutate(const Expression &expr) noexcept;

    virtual ~BaseMutator() = default;
};

}

#endif