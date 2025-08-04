// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef INITIALIZER_BASE_HPP
#define INITIALIZER_BASE_HPP

#include "../../expressions/expression.hpp"

#include <vector>
class BaseInitializer {
public:
    virtual void initialize(std::vector<Expression>& population);

    virtual ~BaseInitializer() = default;
};

#endif