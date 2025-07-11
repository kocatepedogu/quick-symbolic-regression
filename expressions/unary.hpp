// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef UNARY_HPP
#define UNARY_HPP

#include "expression.hpp"

static inline Expression Sin(const Expression &e) {
    // Remove unary operations involving only trainable parameters or constants
    if (e.operation == PARAMETER || e.operation == CONSTANT) {
        return e;
    }

    return Expression(SINE, e);
}

static inline Expression Cos(const Expression &e) {
    // Remove unary operations involving only trainable parameters or constants
    if (e.operation == PARAMETER || e.operation == CONSTANT) {
        return e;
    }

    return Expression(COSINE, e);
}

static inline Expression Exp(const Expression &e) {
    // Remove unary operations involving only trainable parameters or constants
    if (e.operation == PARAMETER || e.operation == CONSTANT) {
        return e;
    }

    return Expression(EXPONENTIAL, e);
}

#endif