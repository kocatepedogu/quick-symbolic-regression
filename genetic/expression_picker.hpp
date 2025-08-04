// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef EXPRESSION_PICKER_HPP
#define EXPRESSION_PICKER_HPP

#include "../expressions/expression.hpp"

namespace qsr {

class ExpressionPicker {
public:
    Expression& pick(Expression &expr) noexcept;

private:
    void pick(Expression *node, Expression **result, int& current, int target) noexcept;
};

}

#endif