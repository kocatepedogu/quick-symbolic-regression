// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "expressions/binary.hpp"
#include "expressions/expression.hpp"
#include "util/rng.hpp"

namespace qsr {
    
Expression operator + (const Expression& left_operand, const Expression& right_operand) noexcept {
    return Expression(ADDITION, left_operand, right_operand);
}

Expression operator - (const Expression& left_operand, const Expression& right_operand) noexcept {
    return Expression(SUBTRACTION, left_operand, right_operand);
}

Expression operator * (const Expression& left_operand, const Expression& right_operand) noexcept {
    return Expression(MULTIPLICATION, left_operand, right_operand);
}

Expression operator / (const Expression& left_operand, const Expression& right_operand) noexcept {
    return Expression(DIVISION, left_operand, right_operand);
}

}