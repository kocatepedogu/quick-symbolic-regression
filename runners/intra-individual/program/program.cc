// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "program.hpp"

#include "../../../compiler/ir.hpp"
#include "../../../compiler/compiler.hpp"

#include <hip/hip_runtime.h>

namespace qsr::intra_individual {
   Program::Program(const std::vector<Expression>& exp_pop) {
        const int num_of_individuals = exp_pop.size();
        this->num_of_individuals = num_of_individuals;

        // Create array for storing number of instructions in each program
        this->num_of_instructions = Array1D<int>(num_of_individuals);

        // Create array for storing stack requirements for each program
        this->stack_req = Array1D<int>(num_of_individuals);

        // Create array for storing intermediate requirements for each program
        this->intermediate_req = Array1D<int>(num_of_individuals);

        // Compile all expressions to IR
        std::vector<IntermediateRepresentation> irs(num_of_individuals);
        for (int i = 0; i < num_of_individuals; ++i) {
            const auto &ir = irs[i] = compile(exp_pop[i]);
            num_of_instructions.ptr[i] = ir.bytecode.size();
            stack_req.ptr[i] = ir.stack_requirement;
            intermediate_req.ptr[i] = ir.intermediate_requirement;
        }

        // Find the longest IR to determine the maximum number of instructions
        int max_num_of_instructions = 0;
        for (const auto& ir : irs) {
            if (ir.bytecode.size() > max_num_of_instructions) {
                max_num_of_instructions = ir.bytecode.size();
            }
        }

        // Create array for storing pointers to individual programs
        this->bytecode = Array2D<Instruction>(num_of_individuals, max_num_of_instructions);

        // Compile every expression to IR and copy to GPU memory
        for (int i = 0; i < num_of_individuals; ++i) {
            // Compile
            const IntermediateRepresentation& ir = irs[i];
            const int num_of_instructions = ir.bytecode.size();
            
            // Copy program contents to GPU
            memcpy(this->bytecode.ptr[i], ir.bytecode.data(), num_of_instructions * sizeof *ir.bytecode.data());
        }
    }
}