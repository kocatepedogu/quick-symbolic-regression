// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "compiler/optimizer.hpp"

#include "expressions/binary.hpp"
#include "expressions/expression.hpp"

#include "util/rng.hpp"

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


namespace qsr {

static Expression optimizedAdd(const Expression& left_operand, const Expression& right_operand) noexcept {
    if (right_operand.operation == CONSTANT && right_operand.value == 0) {
        return left_operand;
    }

    if (left_operand.operation == CONSTANT && left_operand.value == 0) {
        return right_operand;
    }

    if (left_operand.operation == CONSTANT && right_operand.operation == CONSTANT) {
        return left_operand.value + right_operand.value;
    }

    CONSTANT_PROPAGATION(ADDITION, +);
    CONSTANT_PROPAGATION_INVERSE(+, -, SUBTRACTION);

    return Expression(ADDITION, left_operand, right_operand);
}

static Expression optimizedSub(const Expression& left_operand, const Expression& right_operand) noexcept {
    if (right_operand.operation == CONSTANT && right_operand.value == 0) {
        return left_operand;
    }

    if (left_operand == right_operand) {
        return 0.0;
    }

    if (left_operand.operation == CONSTANT && right_operand.operation == CONSTANT) {
        return left_operand.value - right_operand.value;
    }

    return Expression(SUBTRACTION, left_operand, right_operand);
}

static Expression optimizedMul(const Expression& left_operand, const Expression& right_operand) noexcept {
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

    if (left_operand.operation == CONSTANT && right_operand.operation == CONSTANT) {
        return left_operand.value * right_operand.value;
    }

    CONSTANT_PROPAGATION(MULTIPLICATION, *)
    CONSTANT_PROPAGATION_INVERSE(*, /, DIVISION);

    return Expression(MULTIPLICATION, left_operand, right_operand);
}

static Expression optimizedDiv(const Expression& left_operand, const Expression& right_operand) noexcept {
    if (left_operand.operation == CONSTANT && left_operand.value == 0) {
        return 0.0;
    }

    if (right_operand.operation == CONSTANT && right_operand.value == 1) {
        return left_operand;
    }

    if (left_operand == right_operand) {
        return 1.0;
    }

    if (left_operand.operation == CONSTANT && right_operand.operation == CONSTANT) {
        return left_operand.value / right_operand.value;
    }

    return Expression(DIVISION, left_operand, right_operand);
}

static Expression optimizedSin(const Expression &e) {
    if (e.operation == CONSTANT) {
        return sin(e.value);
    }

    return Expression(SINE, e);
}

static Expression optimizedCos(const Expression &e) {
    if (e.operation == CONSTANT) {
        return cos(e.value);
    }

    return Expression(COSINE, e);
}

static Expression optimizedExp(const Expression &e) {
    if (e.operation == CONSTANT) {
        return exp(e.value);
    }

    return Expression(EXPONENTIAL, e);
}

static Expression optimizedReLU(const Expression &e) {
    if (e.operation == CONSTANT) {
        return e.value > 0 ? e.value : 0;
    }

    return Expression(RECTIFIED_LINEAR_UNIT, e);
}

Expression optimize(const Expression &e) {
    switch (e.operation) {
    case CONSTANT:
        return e;
    case IDENTITY:
        return e;
    case PARAMETER:
        return e;
    case ADDITION:
        return optimizedAdd(optimize(e.operands[0]), optimize(e.operands[1]));
    case SUBTRACTION:
        return optimizedSub(optimize(e.operands[0]), optimize(e.operands[1]));
    case MULTIPLICATION:
        return optimizedMul(optimize(e.operands[0]), optimize(e.operands[1]));
    case DIVISION:
        return optimizedDiv(optimize(e.operands[0]), optimize(e.operands[1]));
    case SINE:
        return optimizedSin(optimize(e.operands[0]));
    case COSINE:
        return optimizedCos(optimize(e.operands[0]));
    case RECTIFIED_LINEAR_UNIT:
        return optimizedReLU(optimize(e.operands[0]));
    case EXPONENTIAL:
        return optimizedExp(optimize(e.operands[0]));
    default:
        throw std::runtime_error("Optimizer: Unknown operation");
    }
}

}