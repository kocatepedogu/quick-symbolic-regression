// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef SELECTOR_BASE_HPP
#define SELECTOR_BASE_HPP

#include "../../../expressions/expression.hpp"

namespace qsr {

class BaseSelector {
public:
    virtual void update(const Expression population[]) = 0;

    virtual const Expression& select(const Expression population[]) = 0;

    virtual ~BaseSelector();
};

}

#endif