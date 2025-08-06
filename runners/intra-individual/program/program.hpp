// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef INTRA_PROGRAM_HPP
#define INTRA_PROGRAM_HPP

#include "../../../compiler/ir.hpp"
#include "../../../expressions/expression.hpp"

#include "../../../util/arrays/array1d.hpp"
#include "../../../util/arrays/array2d.hpp"

#include <vector>

namespace qsr::intra_individual {
    struct Program {
        /// Programs
        Array2DF<Instruction> bytecode;

        /// Length of Programs
        Array1D<int> num_of_instructions;

        /// Total number of individuals
        int num_of_individuals;

        /// Constructor
        Program(const std::vector<Expression>& exp_pop);

        /// Default constructor
        Program() = default;
    };
}

#endif