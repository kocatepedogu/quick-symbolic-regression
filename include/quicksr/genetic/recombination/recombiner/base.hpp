// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef RECOMBINER_BASE_HPP_DEFINED
#define RECOMBINER_BASE_HPP_DEFINED

#include "expressions/expression.hpp"

#include <tuple>

namespace qsr {
    class BaseRecombiner {
    public:
        virtual std::tuple<Expression, Expression> recombine(Expression e1, Expression e2) noexcept;

        virtual ~BaseRecombiner() = default;
    };
}

#endif