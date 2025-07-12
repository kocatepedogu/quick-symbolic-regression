// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "bytecode.hpp"

#include <hip/hip_runtime.h>
#include "../util.hpp"

Program::Program(int length) : length(length) {
    HIP_CALL(hipMallocManaged(&bytecode, sizeof *bytecode * length));
    for (int i = 0; i < length; ++i) {
        bytecode[i] = Instruction();
    }
}

Program::~Program() {
    HIP_CALL(hipFree(bytecode));
}

std::ostream& operator << (std::ostream& os, const Instruction& instruction) noexcept {
    switch (instruction.opcode) {
        case PUSH_IMMEDIATE:
            os << "push " << instruction.value;
            break;
        case PUSH_VARIABLE:
            os << "push variable " << instruction.argindex;
            break;
        case PUSH_PARAMETER:
            os << "push parameter " << instruction.argindex;
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
        case NOP:
            os << "nop";
            break;
        case LOSS:
            os << "loss";
            break;
    }

    return os;
}

std::ostream& operator << (std::ostream& os, const Program& program) noexcept {
    for (int i = 0; i < program.length; ++i) {
        os << program.bytecode[i] << std::endl;
    }

    return os;
}
