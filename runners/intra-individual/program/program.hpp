// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef INTRA_PROGRAM_HPP
#define INTRA_PROGRAM_HPP

#include "../../../compiler/ir.hpp"
#include "../../../expressions/expression.hpp"

#include <vector>

namespace intra_individual {
    struct Program {
        /// Programs
        Instruction **bytecode;

        /// Length of Programs
        int *num_of_instructions;

        /// Total number of individuals
        int num_of_individuals;
    };

    void program_create(Program *prog_pop, const std::vector<Expression>& exp_pop);

    void program_destroy(Program &prog_pop);
}

#endif