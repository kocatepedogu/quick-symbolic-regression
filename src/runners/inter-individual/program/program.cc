// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "runners/inter-individual/program/program.hpp"

#include <hip/hip_runtime.h>
#include <omp.h>

#include "compiler/compiler.hpp"

namespace qsr {
extern int nislands;

namespace inter_individual {

Program::Program(const std::vector<Expression>& exp_pop) {
    this->num_of_individuals = exp_pop.size();

    // Compile all expressions to IR and find maximum size requirements in parallel
    compile_expressions(exp_pop);

    // Copy all programs to GPU memory
    copy_to_gpu_memory();
}

void Program::compile_expressions(const std::vector<Expression>& population) {
    // Pre-resize the results vector to allow for thread-safe indexed access.
    irs.resize(num_of_individuals);

    // Initialize global maximums to zero.
    max_num_of_instructions = 0;
    stack_req = 0;
    intermediate_req = 0;

    omp_set_max_active_levels(2);
#pragma omp parallel num_threads(std::min(omp_get_num_procs() / nislands - 1, 1))
    {
        // Each thread maintains its own local maximums to avoid race conditions.
        int local_max_instr = 0;
        int local_stack_req = 0;
        int local_intermediate_req = 0;

        // The 'dynamic' schedule creates a work queue, allowing threads to "steal"
        // the next available expression. This is ideal for load balancing when
        // compilation times for expressions are uneven.
#pragma omp for schedule(dynamic)
        for (int i = 0; i < num_of_individuals; ++i) {
            // Compile the expression.
            IntermediateRepresentation ir = compile(population[i]);

            // Update thread-local maximums.
            if (ir.bytecode.size() > local_max_instr) {
                local_max_instr = ir.bytecode.size();
            }
            if (ir.stack_requirement > local_stack_req) {
                local_stack_req = ir.stack_requirement;
            }
            if (ir.intermediate_requirement > local_intermediate_req) {
                local_intermediate_req = ir.intermediate_requirement;
            }

            // Store the result directly into its final position. Since each
            // thread works on a unique index 'i', this is thread-safe.
            irs[i] = std::move(ir);
        }

        // After the loop, use a lightweight critical section to safely update
        // the global maximums from each thread's local maximums.
#pragma omp critical
        {
            if (local_max_instr > max_num_of_instructions) {
                max_num_of_instructions = local_max_instr;
            }
            if (local_stack_req > stack_req) {
                stack_req = local_stack_req;
            }
            if (local_intermediate_req > intermediate_req) {
                intermediate_req = local_intermediate_req;
            }
        }
    } // End of parallel region
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

}
}