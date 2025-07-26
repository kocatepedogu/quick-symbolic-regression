// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "binary.hpp"
#include "expression.hpp"
#include "../util/rng.hpp"

#define PROPAGATION_TEMPLATE_INVERSE(OP, INVOP, INVERSE_OPERATION, TYPE) \
    if (left_operand.operation == CONSTANT && right_operand.operation == INVERSE_OPERATION) { \
        if (right_operand.operands[0].operation == CONSTANT) { \
            return Expression(left_operand.value OP right_operand.operands[0].value) INVOP right_operand.operands[1]; \
        } \
        if (right_operand.operands[1].operation == CONSTANT) { \
            return Expression(left_operand.value INVOP right_operand.operands[1].value) OP right_operand.operands[0]; \
        } \
    }


#define CONSTANT_PROPAGATION_INVERSE(OP, INVOP, INVERSE_OPERATION) \
    PROPAGATION_TEMPLATE_INVERSE(OP, INVOP, INVERSE_OPERATION, CONSTANT)


#define PARAMETER_PROPAGATION_INVERSE(OP, INVOP, INVERSE_OPERATION) \
    PROPAGATION_TEMPLATE_INVERSE(OP, INVOP, INVERSE_OPERATION, PARAMETER)


#define PROPAGATION_TEMPLATE_HALF(OP, FIRST_OPERAND, SECOND_OPERAND, OPERATION, TYPE) \
    if (FIRST_OPERAND.operation == TYPE && SECOND_OPERAND.operation == OPERATION) { \
        if (SECOND_OPERAND.operands[0].operation == TYPE) { \
            return Expression(FIRST_OPERAND.value OP SECOND_OPERAND.operands[0].value) OP SECOND_OPERAND.operands[1]; \
        } \
        if (SECOND_OPERAND.operands[1].operation == TYPE) { \
            return Expression(FIRST_OPERAND.value OP SECOND_OPERAND.operands[1].value) OP SECOND_OPERAND.operands[0]; \
        } \
    }

#define PROPAGATION_TEMPLATE(OPERATION, OP, TYPE) \
    PROPAGATION_TEMPLATE_HALF(OP, left_operand, right_operand, OPERATION, TYPE) \
    PROPAGATION_TEMPLATE_HALF(OP, right_operand, left_operand, OPERATION, TYPE) \


#define CONSTANT_PROPAGATION(OPERATION, OP) \
    PROPAGATION_TEMPLATE(OPERATION, OP, CONSTANT)


#define PARAMETER_PROPAGATION(OPERATION, OP) \
    PROPAGATION_TEMPLATE(OPERATION, OP, PARAMETER)

    
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

    // Replace binary operations involving both a constant and a trainable parameter with 
    // just a trainable parameter
    if (left_operand.operation == PARAMETER && right_operand.operation == CONSTANT) {
        return left_operand;
    }
    if (left_operand.operation == CONSTANT && right_operand.operation == PARAMETER) {
        return right_operand;
    }

    // Constant propagation and trainable parameter propagation
    CONSTANT_PROPAGATION(ADDITION, +);
    PARAMETER_PROPAGATION(ADDITION, +);
    CONSTANT_PROPAGATION_INVERSE(+, -, SUBTRACTION);
    PARAMETER_PROPAGATION_INVERSE(+, -, SUBTRACTION);
    
    if (thread_local_rng() % 2 == 0) {
        return Expression(ADDITION, left_operand, right_operand);
    }
    else {
        return Expression(ADDITION, right_operand, left_operand);
    }
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

    // Replace binary operations involving both a constant and a trainable parameter with 
    // just a trainable parameter
    if (left_operand.operation == PARAMETER && right_operand.operation == CONSTANT) {
        return left_operand;
    }
    if (left_operand.operation == CONSTANT && right_operand.operation == PARAMETER) {
        return right_operand;
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

    // Replace binary operations involving both a constant and a trainable parameter with 
    // just a trainable parameter
    if (left_operand.operation == PARAMETER && right_operand.operation == CONSTANT) {
        return left_operand;
    }
    if (left_operand.operation == CONSTANT && right_operand.operation == PARAMETER) {
        return right_operand;
    }

    // Constant propagation and trainable parameter propagation
    CONSTANT_PROPAGATION(MULTIPLICATION, *)
    PARAMETER_PROPAGATION(MULTIPLICATION, *)
    CONSTANT_PROPAGATION_INVERSE(*, /, DIVISION);
    PARAMETER_PROPAGATION_INVERSE(*, /, DIVISION);

    if (thread_local_rng() % 2 == 0) {
        return Expression(MULTIPLICATION, left_operand, right_operand);
    }
    else {
        return Expression(MULTIPLICATION, right_operand, left_operand);
    }
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

    // Replace binary operations involving both a constant and a trainable parameter with 
    // just a trainable parameter
    if (left_operand.operation == PARAMETER && right_operand.operation == CONSTANT) {
        return left_operand;
    }
    if (left_operand.operation == CONSTANT && right_operand.operation == PARAMETER) {
        return right_operand;
    }

    return Expression(DIVISION, left_operand, right_operand);
}
