// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "compiler.hpp"
#include "bytecode.hpp"

#include <iostream>

static void compile(const Expression& e, Program& program) noexcept {
    switch (e.operation) {
        case CONSTANT:
            program.bytecode[program.length++] = Instruction(PUSH_IMMEDIATE, e.value);
            break;
        case IDENTITY:
            program.bytecode[program.length++] = Instruction(PUSH_VARIABLE, e.argindex);
            break;
        case PARAMETER:
            program.bytecode[program.length++] = Instruction(PUSH_PARAMETER, e.argindex);
            break;
        case ADDITION:
            compile(e.operands[0], program);
            compile(e.operands[1], program);
            program.bytecode[program.length++] = Instruction(ADD);
            break;
        case SUBTRACTION:
            compile(e.operands[0], program);
            compile(e.operands[1], program);
            program.bytecode[program.length++] = Instruction(ADD);
            break;
        case MULTIPLICATION:
            compile(e.operands[0], program);
            compile(e.operands[1], program);
            program.bytecode[program.length++] = Instruction(MUL);
            break;
        case DIVISION:
            compile(e.operands[0], program);
            compile(e.operands[1], program);
            program.bytecode[program.length++] = Instruction(DIV);
            break;
        case SINE:
            compile(e.operands[0], program);
            program.bytecode[program.length++] = Instruction(SIN);
            break;
        case COSINE:
            compile(e.operands[0], program);
            program.bytecode[program.length++] = Instruction(COS);
            break;
        case EXPONENTIAL:
            compile(e.operands[0], program);
            program.bytecode[program.length++] = Instruction(EXP);
            break;
    }

    if (program.length > max_program_size) {
        std::cerr << "Compiled program exceeded maximum length" << std::endl;
        abort();
    }
}

Program compile(const Expression& e) noexcept {
    Program program;
    compile(e, program);
    return program;
}