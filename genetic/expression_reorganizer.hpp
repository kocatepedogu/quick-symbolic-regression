// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef EXPRESSION_REORGANIZER_HPP
#define EXPRESSION_REORGANIZER_HPP

#include "../expressions/expression.hpp"

namespace qsr {

class ExpressionReorganizer {
public:
    Expression reorganize(const Expression &expr) const noexcept;
};

} // namespace qsr

#endif // EXPRESSION_REORGANIZER_HPP