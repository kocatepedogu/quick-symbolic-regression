// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "program.hpp"

#include <hip/hip_runtime.h>

#include "../../../compiler/compiler.hpp"

using namespace qsr::inter_individual;

Program::Program(const std::vector<Expression>& exp_pop) {
    this->num_of_individuals = exp_pop.size();

    // Compile all expressions to IR and find maximum size requirements
    compile_expressions(exp_pop);

    // Copy all programs to GPU memory
    copy_to_gpu_memory();
}

void Program::compile_expressions(const std::vector<Expression>& population) {
    this->max_num_of_instructions = 0;
    this->stack_req = 0;
    this->intermediate_req = 0;

    // Compile each expression to IR and determine the longest IR length
    for (int i = 0; i < num_of_individuals; ++i) {
        const IntermediateRepresentation& ir = compile(population[i]);

        // Largest number of instructions (to determine pad length)
        if (ir.bytecode.size() > this->max_num_of_instructions) {
            this->max_num_of_instructions = ir.bytecode.size();
        }

        // Largest number of stack elements required (to determine pad length)
        if (ir.stack_requirement > this->stack_req) {
            this->stack_req = ir.stack_requirement;
        }

        // Largest number of intermediate elements required (to determine pad length)
        if (ir.intermediate_requirement > this->intermediate_req) {
            this->intermediate_req = ir.intermediate_requirement;
        }

        irs.push_back(ir);
    }
}

void Program::copy_to_gpu_memory() {
    // Allocate bytecode array on GPU
    this->bytecode = Array2D<Instruction>(this->max_num_of_instructions, num_of_individuals);

    // Copy instructions to GPU memory
    for (int j = 0; j < num_of_individuals; ++j) {
        const IntermediateRepresentation& ir = irs[j];

        for (int i = 0; i < this->max_num_of_instructions; ++i) {
            if (i < ir.bytecode.size()) {
                // Copy from IR
                this->bytecode.ptr[i][j] = ir.bytecode[i];
            } else {
                // Pad with NOP at the end
                this->bytecode.ptr[i][j] = Instruction();
            }
        }
    }
}
