// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "program.hpp"

#include <hip/hip_runtime.h>

#include "../../compiler/compiler.hpp"
#include "../../util.hpp"

namespace inter_individual {
    void program_create(Program *prog_pop, const std::vector<Expression>& exp_pop) {
        const int num_of_individuals = exp_pop.size();
        prog_pop->num_of_individuals = num_of_individuals;

        // Intermediate representation of each expression
        std::vector<IntermediateRepresentation> ir_list;

        // Largest number of instructions (to determine pad length)
        prog_pop->max_num_of_instructions = 0;

        // Compile each expression to IR and determine the longest IR length
        for (int i = 0; i < num_of_individuals; ++i) {
            const IntermediateRepresentation& ir = compile(exp_pop[i]);
            const int num_of_instructions = ir.bytecode.size();

            ir_list.push_back(ir);
            prog_pop->max_num_of_instructions = num_of_instructions > prog_pop->max_num_of_instructions ? 
                num_of_instructions : prog_pop->max_num_of_instructions;
        }

        // Element i points to the array containing ith instructions of every program
        HIP_CALL(hipMallocManaged(&prog_pop->bytecode, 
            sizeof *prog_pop->bytecode * prog_pop->max_num_of_instructions));
        for (int i = 0; i < prog_pop->max_num_of_instructions; ++i) {
            HIP_CALL(hipMallocManaged(&prog_pop->bytecode[i], sizeof prog_pop->bytecode[0] * num_of_individuals));
        }

        // Copy instructions to GPU memory
        for (int j = 0; j < num_of_individuals; ++j) {
            // Compile
            const IntermediateRepresentation& ir = ir_list[j];
            const int num_of_instructions = ir.bytecode.size();
            
            for (int i = 0; i < prog_pop->max_num_of_instructions; ++i) {
                if (i < num_of_instructions) {
                    // Copy from IR
                    prog_pop->bytecode[i][j] = ir.bytecode[i];
                } else {
                    // Pad with NOP at the end
                    prog_pop->bytecode[i][j] = Instruction();
                }
            }
        }
    }

    void program_destroy(Program &prog_pop) {
        for (int i = 0; i < prog_pop.max_num_of_instructions; ++i) {
            HIP_CALL(hipFree(prog_pop.bytecode[i]));
        }
        HIP_CALL(hipFree(prog_pop.bytecode));
    }
}
