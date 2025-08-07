// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "program.hpp"

#include <hip/hip_runtime.h>

#include "../../../compiler/compiler.hpp"

namespace qsr::inter_individual {
    Program::Program(const std::vector<Expression>& exp_pop) {
        const int num_of_individuals = exp_pop.size();
        this->num_of_individuals = num_of_individuals;

        // Intermediate representation of each expression
        std::vector<IntermediateRepresentation> ir_list;

        // Largest number of instructions (to determine pad length)
        this->max_num_of_instructions = 0;

        // Largest number of stack elements required
        this->stack_req = 0;

        // Largest number of intermediate elements required
        this->intermediate_req = 0;

        // Compile each expression to IR and determine the longest IR length
        for (int i = 0; i < num_of_individuals; ++i) {
            const IntermediateRepresentation& ir = compile(exp_pop[i]);
            const int num_of_instructions = ir.bytecode.size();

            ir_list.push_back(ir);

            if (num_of_instructions > this->max_num_of_instructions) {
                this->max_num_of_instructions = num_of_instructions;
            }

            if (ir.stack_requirement > this->stack_req) {
                this->stack_req = ir.stack_requirement;
            }

            if (ir.intermediate_requirement > this->intermediate_req) {
                this->intermediate_req = ir.intermediate_requirement;
            }
        }

        // Element i points to the array containing ith instructions of every program
        this->bytecode = Array2D<Instruction>(this->max_num_of_instructions, num_of_individuals);

        // Copy instructions to GPU memory
        for (int j = 0; j < num_of_individuals; ++j) {
            // Compile
            const IntermediateRepresentation& ir = ir_list[j];
            const int num_of_instructions = ir.bytecode.size();
            
            for (int i = 0; i < this->max_num_of_instructions; ++i) {
                if (i < num_of_instructions) {
                    // Copy from IR
                    this->bytecode.ptr[i][j] = ir.bytecode[i];
                } else {
                    // Pad with NOP at the end
                    this->bytecode.ptr[i][j] = Instruction();
                }
            }
        }
    }
}