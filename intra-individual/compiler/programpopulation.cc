#include "programpopulation.hpp"

#include "../../compiler/ir.hpp"
#include "../../compiler/compiler.hpp"
#include "../../util.hpp"

#include <hip/hip_runtime.h>

namespace intra_individual {
    void program_create(ProgramPopulation *prog_pop, const std::vector<Expression>& exp_pop) {
        const int num_of_individuals = exp_pop.size();
        prog_pop->num_of_individuals = num_of_individuals;

        // Create array for storing number of instructions in each program
        HIP_CALL(hipMallocManaged(&prog_pop->num_of_instructions, 
            sizeof *prog_pop->num_of_instructions * num_of_individuals));

        // Create array for storing pointers to individual programs
        HIP_CALL(hipMallocManaged(&prog_pop->bytecode, 
            sizeof *prog_pop->bytecode * num_of_individuals));

        // Compile every expression to IR and copy to GPU memory
        for (int i = 0; i < num_of_individuals; ++i) {
            // Compile
            const IntermediateRepresentation& ir = compile(exp_pop[i]);
            const int num_of_instructions = ir.bytecode.size();

            // Copy program size to GPU
            prog_pop->num_of_instructions[i] = num_of_instructions;
            
            // Copy program contents to GPU
            HIP_CALL(hipMallocManaged(&prog_pop->bytecode[i], sizeof *prog_pop->bytecode[i] * num_of_instructions));
            memcpy(prog_pop->bytecode[i], &ir.bytecode[0], num_of_instructions * sizeof ir.bytecode[0]);
        }
    }

    void program_destroy(ProgramPopulation &prog_pop) {
        // Delete program contents from GPU memory
        for (int i = 0; i < prog_pop.num_of_individuals; ++i) {
            HIP_CALL(hipFree(prog_pop.bytecode[i]));
        }

        // Delete array for storing pointers to individual programs
        HIP_CALL(hipFree(prog_pop.bytecode));

        // Delete array for storing number of instructions in each program
        HIP_CALL(hipFree(prog_pop.num_of_instructions));
    }
}