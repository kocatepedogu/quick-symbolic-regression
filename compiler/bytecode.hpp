// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef BYTECODE_HPP
#define BYTECODE_HPP

#include <ostream>

enum Opcode {
    PUSH_IMMEDIATE,
    PUSH_VARIABLE,
    PUSH_PARAMETER,
    ADD,
    SUB,
    MUL,
    DIV,
    SIN,
    COS,
    EXP,
    NOP,
    LOSS
};

struct Instruction {
    Opcode opcode;

    union {
        /// For instructions with immediate operand
        float value;

        /// For instructions with index operand
        int argindex;
    };

    // For backpropagation, at which intermediate array location the inputs to
    // this operation are stored. In forward propagation, this is set to negative one.
    int intermediate_index;

    /** @brief No operation */
    constexpr Instruction() :
        opcode(NOP), intermediate_index(-1) {}

    /** @brief Instructions with immediate operand */
    constexpr Instruction(Opcode opcode, float value) :
        opcode(opcode), value(value), intermediate_index(-1) {}

    /** @brief Instructions with index operand */
    constexpr Instruction(Opcode opcode, int argindex) :
        opcode(opcode), argindex(argindex), intermediate_index(-1) {}

    /** @brief Instructions whose operands are entirely stored on the stack */
    constexpr Instruction(Opcode opcode) :
        opcode(opcode), intermediate_index(-1) {}
};


struct Program {
    /// Instructions
    Instruction *bytecode;

    /// Total number of instructions
    int length;
    
    /** @brief Creates empty program on GPU memory */
    Program(int length);

    /** @brief Deletes program from GPU memory */
    ~Program();
};

/**
  * @brief Yields text representation of a single instruction
  */
std::ostream& operator << (std::ostream& os, const Instruction& instruction) noexcept;

/**
  * @brief Yields text representation of a program
  */
std::ostream& operator << (std::ostream& os, const Program& program) noexcept;

#endif