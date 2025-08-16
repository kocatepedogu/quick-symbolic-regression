// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include <string>
#include <vector>
#include <cassert>

#include "expression.hpp"

namespace qsr {

Expression::Expression(operation_t operation, const Expression& e) noexcept :
    operation(operation), 
    operands({e}), 
    num_of_nodes(e.num_of_nodes + 1),
    depth(e.depth + 1) {}

    
Expression::Expression(operation_t operation, const Expression& e1, const Expression& e2) noexcept :
    operation(operation), 
    operands({e1, e2}), 
    num_of_nodes(e1.num_of_nodes + e2.num_of_nodes + 1),
    depth(std::max(e1.depth, e2.depth) + 1) {}


static std::string to_string(const Expression& node, const Expression& root) {
    std::string result;

    switch (node.operation) {
        case CONSTANT:
            result = std::to_string(node.value);
            break;
        case IDENTITY:
            result = "x" + std::to_string(node.argindex);
            break;
        case PARAMETER:
            result = "w" + std::to_string(node.argindex);
            if (!root.weights.empty())
                result += "=" + std::to_string(root.weights[node.argindex]);
            break;
        case ADDITION:
            result = "(" + to_string(node.operands[0], root) + ") + (" + to_string(node.operands[1], root) + ")";
            break;
        case SUBTRACTION:
            result = "(" + to_string(node.operands[0], root) + ") - (" + to_string(node.operands[1], root) + ")";
            break;
        case MULTIPLICATION:
            result = "(" + to_string(node.operands[0], root) + ") * (" + to_string(node.operands[1], root) + ")";
            break;
        case DIVISION:
            result = "(" + to_string(node.operands[0], root) + ") / (" + to_string(node.operands[1], root) + ")";
            break;
        case SINE:
            result = "sin(" + to_string(node.operands[0], root) + ")";
            break;
        case COSINE:
            result = "cos(" + to_string(node.operands[0], root) + ")";
            break;
        case EXPONENTIAL:
            result = "exp(" + to_string(node.operands[0], root) + ")";
            break;
        case RECTIFIED_LINEAR_UNIT:
            result = "relu(" + to_string(node.operands[0], root) + ")";
            break;
    }

    return result;
}


std::ostream& operator<<(std::ostream& os, const Expression& e) noexcept
{
    return os << to_string(e, e);
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

bool operator < (const Expression& left_operand, const Expression& right_operand) noexcept {
    return left_operand.fitness < right_operand.fitness;
}

bool operator > (const Expression& left_operand, const Expression& right_operand) noexcept {
    return left_operand.fitness > right_operand.fitness;
}

}