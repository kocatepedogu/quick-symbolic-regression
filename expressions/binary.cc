// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "binary.hpp"

Expression operator + (const Expression& left_operand, const Expression& right_operand) noexcept {
    if (right_operand.operation == CONSTANT && right_operand.value == 0) {
        return left_operand;
    }

    if (left_operand.operation == CONSTANT && left_operand.value == 0) {
        return right_operand;
    }

    // Remove binary operations involving only trainable parameters
    if (left_operand.operation == PARAMETER && right_operand.operation == PARAMETER) {
        return left_operand;
    }

    // Remove binary operations involving only constants
    if (left_operand.operation == CONSTANT && right_operand.operation == CONSTANT) {
        return left_operand.value + right_operand.value;
    }

    return Expression(ADDITION, left_operand, right_operand);
}

Expression operator - (const Expression& left_operand, const Expression& right_operand) noexcept {
    if (right_operand.operation == CONSTANT && right_operand.value == 0) {
        return left_operand;
    }

    if (left_operand == right_operand) {
        return 0.0;
    }

    // Remove binary operations involving only trainable parameters
    if (left_operand.operation == PARAMETER && right_operand.operation == PARAMETER) {
        return left_operand;
    }

    // Remove binary operations involving only constants
    if (left_operand.operation == CONSTANT && right_operand.operation == CONSTANT) {
        return left_operand.value - right_operand.value;
    }

    return Expression(SUBTRACTION, left_operand, right_operand);
}

Expression operator * (const Expression& left_operand, const Expression& right_operand) noexcept {
    if (right_operand.operation == CONSTANT && right_operand.value == 0) {
        return 0.0;
    }

    if (left_operand.operation == CONSTANT && left_operand.value == 0) {
        return 0.0;
    }

    if (right_operand.operation == CONSTANT && right_operand.value == 1) {
        return left_operand;
    }

    if (left_operand.operation == CONSTANT && left_operand.value == 1) {
        return right_operand;
    }

    // Remove binary operations involving only trainable parameters
    if (left_operand.operation == PARAMETER && right_operand.operation == PARAMETER) {
        return left_operand;
    }

    // Remove binary operations involving only constants
    if (left_operand.operation == CONSTANT && right_operand.operation == CONSTANT) {
        return left_operand.value * right_operand.value;
    }

    return Expression(MULTIPLICATION, left_operand, right_operand);
}

Expression operator / (const Expression& left_operand, const Expression& right_operand) noexcept {
    // Replace all division by zero by the left operand. This type of mutation is useless.
    if (right_operand.operation == CONSTANT && right_operand.value == 0) {
        return left_operand;
    }

    if (left_operand.operation == CONSTANT && left_operand.value == 0) {
        return 0.0;
    }

    if (right_operand.operation == CONSTANT && right_operand.value == 1) {
        return left_operand;
    }

    if (left_operand == right_operand) {
        return 1.0;
    }

    // Remove binary operations involving only trainable parameters
    if (left_operand.operation == PARAMETER && right_operand.operation == PARAMETER) {
        return left_operand;
    }

    // Remove binary operations involving only constants
    if (left_operand.operation == CONSTANT && right_operand.operation == CONSTANT) {
        return left_operand.value / right_operand.value;
    }

    return Expression(DIVISION, left_operand, right_operand);
}
