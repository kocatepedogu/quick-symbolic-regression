// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef UNARY_HPP
#define UNARY_HPP

#include "expression.hpp"

namespace qsr {

static inline Expression Sin(const Expression &e) {
    return Expression(SINE, e);
}

static inline Expression Cos(const Expression &e) {
    return Expression(COSINE, e);
}

static inline Expression Exp(const Expression &e) {
    return Expression(EXPONENTIAL, e);
}

static inline Expression ReLU(const Expression &e) {
    return Expression(RECTIFIED_LINEAR_UNIT, e);
}

}

#endif