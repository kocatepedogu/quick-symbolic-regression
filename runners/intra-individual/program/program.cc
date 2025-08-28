// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "program.hpp"

#include "compiler/ir.hpp"
#include "compiler/compiler.hpp"

#include <hip/hip_runtime.h>

using namespace qsr::intra_individual;

Program::Program(const std::vector<Expression>& population) {
    const int num_of_individuals = population.size();

    // Allocate arrays for storing number of instructions, stack requirements, 
    // and intermediate requirements for every program
    allocate_arrays(num_of_individuals);

    // Compile all expressions to intermediate representation
    compile_expressions(population);

    // Find the longest IR to determine the maximum number of instructions
    find_longest_ir();

    // Copy all programs to GPU memory
    copy_to_gpu_memory();
}

void Program::allocate_arrays(const int num_of_individuals) {
    this->num_of_individuals = num_of_individuals;

    // Create array for storing number of instructions in each program
    this->num_of_instructions = Array1D<int>(num_of_individuals);

    // Create array for storing stack requirements of each program
    this->stack_req = Array1D<int>(num_of_individuals);

    // Create array for storing intermediate requirements of each program
    this->intermediate_req = Array1D<int>(num_of_individuals);
}

void Program::compile_expressions(const std::vector<Expression>& population) {
    // Resize the IRs vector to accommodate all individuals
    irs.resize(num_of_individuals);

    // Loop through each expression in the population
    for (int i = 0; i < num_of_individuals; ++i) {
        // Compile each expression to intermediate representation
        const auto &ir = irs[i] = compile(population[i]);

        // Store the number of instructions, stack requirement, and intermediate requirement for each program
        num_of_instructions.ptr[i] = ir.bytecode.size();
        stack_req.ptr[i] = ir.stack_requirement;
        intermediate_req.ptr[i] = ir.intermediate_requirement;
    }
}

void Program::find_longest_ir() {
    // Find the maximum number of instructions
    max_num_of_instructions = 0;
    for (const auto& ir : irs) {
        if (ir.bytecode.size() > max_num_of_instructions) {
            max_num_of_instructions = ir.bytecode.size();
        }
    }
}

void Program::copy_to_gpu_memory() {
    // Allocate bytecode array on GPU
    this->bytecode = Array2D<Instruction>(num_of_individuals, max_num_of_instructions);

    // Copy instructions from all expressions to GPU memory
    for (int i = 0; i < num_of_individuals; ++i) {     
        auto& ir = irs[i];
        memcpy(this->bytecode.ptr[i], ir.bytecode.data(), ir.bytecode.size() * sizeof *ir.bytecode.data());
    }
}

ProgramIterator Program::begin() {
    return ProgramIterator(this);
}

ProgramIterator Program::end() {
    return ProgramIterator(this, num_of_individuals);
}