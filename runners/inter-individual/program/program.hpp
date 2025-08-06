// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef INTER_PROGRAMPOPULATION_HPP
#define INTER_PROGRAMPOPULATION_HPP

#include "../../../compiler/ir.hpp"
#include "../../../expressions/expression.hpp"

#include "../../../util/arrays/array2d.hpp"

#include <vector>

namespace qsr::inter_individual {
    struct Program {
        /// Programs
        Array2D<Instruction> bytecode;

        /// Length of the longest program
        int max_num_of_instructions;

        /// Total number of individuals
        int num_of_individuals;

        /// Constructor
        Program(const std::vector<Expression>& exp_pop);

        /// Default constructor
        Program() = default;
    };
}

#endif