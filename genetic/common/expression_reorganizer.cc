// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "expression_reorganizer.hpp"

#include "../../../expressions/unary.hpp"
#include "../../../expressions/binary.hpp"

namespace qsr {
    Expression ExpressionReorganizer::reorganize(const Expression& e) const noexcept {
        #define _REORGANIZE_CALL(i) \
            reorganize(e.operands[i])

        switch (e.operation) {
            case CONSTANT:
            case IDENTITY:
            case PARAMETER:
                return e;

            BINARY_OP_CASE(ADDITION, _REORGANIZE_CALL, +);
            BINARY_OP_CASE(SUBTRACTION, _REORGANIZE_CALL, -);
            BINARY_OP_CASE(MULTIPLICATION, _REORGANIZE_CALL, *);
            BINARY_OP_CASE(DIVISION, _REORGANIZE_CALL, /);
            UNARY_OP_CASE(SINE, _REORGANIZE_CALL, Sin);
            UNARY_OP_CASE(COSINE, _REORGANIZE_CALL, Cos);
            UNARY_OP_CASE(EXPONENTIAL, _REORGANIZE_CALL, Exp);
            UNARY_OP_CASE(RECTIFIED_LINEAR_UNIT, _REORGANIZE_CALL, ReLU);
        }
    }
};