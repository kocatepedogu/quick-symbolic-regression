// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "compiler.hpp"
#include "program.hpp"
#include "programpopulation.hpp"


// Intermediate compiler state
struct CompilerState {
    /// At which location forward propagation instructions begin
    int forward_offset;

    /// At which location backpropagation instructions begin
    int backward_offset;

    /// Number of forward propagation instructions
    int forward_length;

    /// Number of backpropagation instructions
    int backward_length;

    /// The expected intermediate stack pointer value just after the 
    /// currently processed forward propagation instruction is executed.
    int expected_sp;
};


// Forward declaration for indirect recursion
static void compile(const Expression& e, Program* p, CompilerState& s) noexcept;


// Generates a single instruction
template <int opcount, typename ...T>
static void compile(const Expression& e, Program *p, CompilerState& s, T... args) {
    // In backpropagation, operation comes first, operands come afterwards
    auto& backprop_inst = p->bytecode[s.backward_offset + s.backward_length++];
    backprop_inst = Instruction(args...);

    // Operands
    #pragma unroll
    for (int i = 0; i < opcount; ++i) {
        compile(e.operands[i], p, s);
    }
    
    // In forward propagation, operands come first, followed by operation
    auto& forwprop_inst = p->bytecode[s.forward_offset + s.forward_length++];
    forwprop_inst = Instruction(args...);

    // Calculate at which point the stack intermediate pointer will be before and after this 
    // forward propagation instruction is executed. Save the expected value before the execution
    // into the memindex field of associated backpropagation instruction.

    backprop_inst.intermediate_index = s.expected_sp;
    s.expected_sp += opcount;
}


static void compile(const Expression& e, Program* p, CompilerState& s) noexcept {
    switch (e.operation) {
        case CONSTANT:
            compile<0>(e, p, s, PUSH_IMMEDIATE, e.value);
            break;
        case IDENTITY:
            compile<0>(e, p, s, PUSH_VARIABLE, e.argindex);
            break;
        case PARAMETER:
            compile<0>(e, p, s, PUSH_PARAMETER, e.argindex);
            break;
        case ADDITION:
            compile<2>(e, p, s, ADD);
            break;
        case SUBTRACTION:
            compile<2>(e, p, s, SUB);
            break;
        case MULTIPLICATION:
            compile<2>(e, p, s, MUL);
            break;
        case DIVISION:
            compile<2>(e, p, s, DIV);
            break;
        case SINE:
            compile<1>(e, p, s, SIN);
            break;
        case COSINE:
            compile<1>(e, p, s, COS);
            break;
        case EXPONENTIAL:
            compile<1>(e, p, s, EXP);
            break;
    }
}

static void compile(const Expression& e, Program* p) noexcept {
    // - Initially, the number of instructions are set to zero.
    // - In the end, there will be e.num_of_nodes different instructions for both,
    //   and one additional instruction in betwen for loss evaluation.
    // - The instructions for forward propagation start at index 0.
    // - The instructions for backpropagation start at index e.num_of_nodes + 1

    CompilerState s;

    s.forward_offset = 0;
    s.backward_offset = e.num_of_nodes + 1;
    s.forward_length = 0;
    s.backward_length = 0;
    s.expected_sp = 0;

    // Compile both forward propagation and backpropagation

    compile(e, p, s);

    // Add loss function evaluation in between forward propagation and backpropagation

    p->bytecode[s.backward_offset - 1] = Instruction(LOSS);
}

Program compile(const Expression& e) noexcept {
    Program p(2*e.num_of_nodes + 1);
    compile(e, &p);
    return p;
}

ProgramPopulation compile(const std::vector<Expression>& exp_pop) noexcept {
    ProgramPopulation programPopulation(exp_pop.size());
    for (int i = 0; i < exp_pop.size(); ++i) {
        programPopulation.individuals[i] = new Program(compile(exp_pop[i]));
    }
    return programPopulation;
}