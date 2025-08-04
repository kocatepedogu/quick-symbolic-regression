// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef MUTATION_BASE_HPP
#define MUTATION_BASE_HPP

#include "../../expressions/expression.hpp"

namespace qsr {

class BaseMutation {
public:
    virtual Expression mutate(const Expression &expr) noexcept;

    virtual ~BaseMutation() = default;
};

}

#endif