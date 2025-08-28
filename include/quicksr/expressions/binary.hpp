// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef BINARY_HPP
#define BINARY_HPP

#include "expressions/expression.hpp"

namespace qsr {

Expression operator + (const Expression& left_operand, const Expression& right_operand) noexcept;

Expression operator - (const Expression& left_operand, const Expression& right_operand) noexcept;

Expression operator * (const Expression& left_operand, const Expression& right_operand) noexcept;

Expression operator / (const Expression& left_operand, const Expression& right_operand) noexcept;

}

#endif