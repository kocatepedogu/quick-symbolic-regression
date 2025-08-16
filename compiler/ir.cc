// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "ir.hpp"

#include <cmath>
#include <iomanip>

namespace qsr {

IntermediateRepresentation::IntermediateRepresentation(int length) : bytecode(length) {
    for (int i = 0; i < length; ++i) {
        bytecode[i] = Instruction();
    }
}

std::ostream& operator << (std::ostream& os, const Instruction& instruction) noexcept {
    if (instruction.intermediate_index >= 0) {
        os << "grad_";
    }

    switch (instruction.opcode) {
        case PUSH_IMMEDIATE:
            os << "imm " << instruction.value;
            break;
        case PUSH_VARIABLE:
            os << "var " << instruction.argindex;
            break;
        case PUSH_PARAMETER:
            os << "param " << instruction.argindex;
            break;
        case ADD:
            os << "add";
            break;
        case SUB:
            os << "sub";
            break;
        case MUL:
            os << "mul";
            break;
        case DIV:
            os << "div";
            break;
        case SIN:
            os << "sin";
            break;
        case COS:
            os << "cos";
            break;
        case EXP:
            os << "exp";
            break;
        case RELU:
            os << "relu";
            break;
        case NOP:
            os << "nop";
            break;
        case LOSS:
            os << "loss";
            break;
    }

    if (instruction.intermediate_index >= 0) {
        os << " [" << instruction.intermediate_index << "]";
    }

    return os;
}

std::ostream& operator << (std::ostream& os, const IntermediateRepresentation& program) noexcept {
    int max_line_number_digits = ceil(log10(program.bytecode.size()));

    os << "Program instructions: " << std::endl;
    for (int i = 0; i < program.bytecode.size(); ++i) {
        os << std::setw(max_line_number_digits) << std::right << std::setfill(' ') << i << " ";
        os << program.bytecode[i] << std::endl;
    }

    return os;
}

}