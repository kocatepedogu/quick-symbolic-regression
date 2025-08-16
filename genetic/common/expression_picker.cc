// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "expression_picker.hpp"

#include "../../util/rng.hpp"
#include "../../util/macro.hpp"

namespace qsr {

Expression& ExpressionPicker::pick(Expression &expr) noexcept {
    int target = thread_local_rng() % expr.num_of_nodes;
    int current = 0;

    Expression *result = nullptr;
    pick(&expr, &result, current, target);
    return *result;
}

void ExpressionPicker::pick(Expression *node, Expression **result, int& current, int target) noexcept {
    if (current++ == target) {
        *result = node;
    }

    #define _PICK_CALL(i) \
        pick(&node->operands[i], result, current, target)

    switch (node->operation) {
        case CONSTANT:
        case IDENTITY:
        case PARAMETER:
            return;

        VOID_BINARY_OP_CASE(ADDITION, _PICK_CALL);
        VOID_BINARY_OP_CASE(SUBTRACTION, _PICK_CALL);
        VOID_BINARY_OP_CASE(MULTIPLICATION, _PICK_CALL);
        VOID_BINARY_OP_CASE(DIVISION, _PICK_CALL);
        VOID_UNARY_OP_CASE(SINE, _PICK_CALL);
        VOID_UNARY_OP_CASE(COSINE, _PICK_CALL);
        VOID_UNARY_OP_CASE(EXPONENTIAL, _PICK_CALL);
    }
}

}