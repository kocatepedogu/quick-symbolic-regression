#include "program.hpp"

#include "../../compiler/ir.hpp"
#include "../../compiler/compiler.hpp"
#include "../../util.hpp"

#include <hip/hip_runtime.h>

namespace intra_individual {
    void program_create(Program *prog_pop, const std::vector<Expression>& exp_pop) {
        const int num_of_individuals = exp_pop.size();
        prog_pop->num_of_individuals = num_of_individuals;

        // Create array for storing number of instructions in each program
        init_arr_1d(prog_pop->num_of_instructions, num_of_individuals);

        // Create array for storing pointers to individual programs
        init_arr_1d(prog_pop->bytecode, num_of_individuals);

        // Compile every expression to IR and copy to GPU memory
        for (int i = 0; i < num_of_individuals; ++i) {
            // Compile
            const IntermediateRepresentation& ir = compile(exp_pop[i]);
            const int num_of_instructions = ir.bytecode.size();

            // Copy program size to GPU
            prog_pop->num_of_instructions[i] = num_of_instructions;
            
            // Copy program contents to GPU
            init_arr_1d(prog_pop->bytecode[i], num_of_instructions);
            memcpy(prog_pop->bytecode[i], &ir.bytecode[0], num_of_instructions * sizeof ir.bytecode[0]);
        }
    }

    void program_destroy(Program &prog_pop) {
        // Delete program contents from GPU memory
        del_arr_2d(prog_pop.bytecode, prog_pop.num_of_individuals);

        // Delete array for storing number of instructions in each program
        del_arr_1d(prog_pop.num_of_instructions);
    }
}