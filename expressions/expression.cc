// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include <vector>
#include <cassert>

#include "expression.hpp"


Expression::Expression(operation_t operation, const Expression& e) noexcept :
    operation(operation), 
    operands({e}), 
    num_of_nodes(e.num_of_nodes + 1) {}

    
Expression::Expression(operation_t operation, const Expression& e1, const Expression& e2) noexcept :
    operation(operation), 
    operands({e1, e2}), 
    num_of_nodes(e1.num_of_nodes + e2.num_of_nodes + 1) {}


std::ostream& operator<<(std::ostream& os, const Expression& e) noexcept
{
    switch (e.operation) {
        case CONSTANT:
            os << e.value;
            break;
        case IDENTITY:
            os << "x" << e.argindex << "";
            break;
        case PARAMETER:
            os << "w" << e.argindex << "";
            break;
        case ADDITION:
            os << "(" << e.operands[0] << ") + (" << e.operands[1] << ")";
            break;
        case SUBTRACTION:
            os << "(" << e.operands[0] << ") - (" << e.operands[1] << ")";
            break;
        case MULTIPLICATION:
            os << "(" << e.operands[0] << ") * (" << e.operands[1] << ")";
            break;
        case DIVISION:
            os << "(" << e.operands[0] << ") / (" << e.operands[1] << ")";
            break;
        case SINE:
            os << "sin(" << e.operands[0] << ")";
            break;
        case COSINE:
            os << "cos(" << e.operands[0] << ")";
            break;
        case EXPONENTIAL:
            os << "exp(" << e.operands[0] << ")";
            break;
    }

    return os;
}

bool operator == (const Expression& left_operand, const Expression& right_operand) noexcept {
    if (left_operand.operation != right_operand.operation) {
        return false;
    }

    if (left_operand.operation == CONSTANT) {
        return left_operand.value == right_operand.value;
    }

    if (left_operand.operation == IDENTITY || left_operand.operation == PARAMETER) {
        return left_operand.argindex == right_operand.argindex;
    }

    assert(left_operand.operands.size() == right_operand.operands.size());
    
    for (int i = 0; i < left_operand.operands.size(); ++i) {
        if (left_operand.operands[i] != right_operand.operands[i]) {
            return false;
        }
    }

    return true;
}

bool operator != (const Expression& left_operand, const Expression& right_operand) noexcept {
    return !(left_operand == right_operand);
}
