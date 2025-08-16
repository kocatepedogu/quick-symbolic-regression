// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "compiler.hpp"
#include "ir.hpp"

namespace qsr {


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

    /// The expected intermediate pointer value just after the 
    /// currently processed forward propagation instruction is executed.
    int expected_ip;

    /// The expected stack pointer value just after the 
    /// currently processed forward propagation instruction is executed.
    int expected_sp_forward;

    /// The expected stack pointer value just after the 
    /// currently processed backpropagation instruction is executed.
    int expected_sp_backward;

    /// The largest stack pointer value ever reached during forward propagation
    int max_sp_forward;

    /// The largest stack pointer value ever reached during backpropagation
    int max_sp_backward;

    /// The largest intermediate pointer value ever reached during forward propagation
    int max_ip;
};


// Forward declaration for indirect recursion
static void compile(const Expression& e, IntermediateRepresentation* p, CompilerState& s) noexcept;


// Generates a single instruction
template <int opcount, typename ...T>
static void compile(const Expression& e, IntermediateRepresentation *p, CompilerState& s, T... args) {
    // In backpropagation, operation comes first, operands come afterwards
    auto& backprop_inst = p->bytecode[s.backward_offset + s.backward_length++];
    backprop_inst = Instruction(args...);

    // Calculate the point at which the stack pointer will be before and after this 
    // backpropagation instruction is executed. Save the expected value before the execution
    s.expected_sp_backward += opcount - 1;
    if (s.expected_sp_backward > s.max_sp_backward) {
        s.max_sp_backward = s.expected_sp_backward;
    }

    // Operands
    #pragma unroll
    for (int i = 0; i < opcount; ++i) {
        compile(e.operands[i], p, s);
    }
    
    // In forward propagation, operands come first, followed by operation
    auto& forwprop_inst = p->bytecode[s.forward_offset + s.forward_length++];
    forwprop_inst = Instruction(args...);

    // Calculate the point at which the intermediate pointer will be before and after this 
    // forward propagation instruction is executed. Save the expected value before the execution
    // into the memindex field of associated backpropagation instruction.

    backprop_inst.intermediate_index = s.expected_ip;
    s.expected_ip += opcount;
    if (s.expected_ip > s.max_ip) {
        s.max_ip = s.expected_ip;
    }

    // Calculate the point at which the stack pointer will be before and after this 
    // forward propagation instruction is executed. Save the expected value after the execution
    s.expected_sp_forward -= opcount - 1;
    if (s.expected_sp_forward > s.max_sp_forward) {
        s.max_sp_forward = s.expected_sp_forward;
    }
}


static void compile(const Expression& e, IntermediateRepresentation* p, CompilerState& s) noexcept {
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
        case RECTIFIED_LINEAR_UNIT:
            compile<1>(e, p, s, RELU);
            break;
        case EXPONENTIAL:
            compile<1>(e, p, s, EXP);
            break;
    }
}

static void compile(const Expression& e, IntermediateRepresentation* p) noexcept {
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
    s.expected_ip = 0;
    s.expected_sp_forward = 0;
    s.expected_sp_backward = 0;
    s.max_sp_forward = 0;
    s.max_sp_backward = 1;
    s.max_ip = 0;

    // Compile both forward propagation and backpropagation
    compile(e, p, s);

    // Add loss function evaluation in between forward propagation and backpropagation
    p->bytecode[s.backward_offset - 1] = Instruction(LOSS);

    // Save the stack size required for running this program
    p->stack_requirement = std::max(s.max_sp_forward, s.max_sp_backward) + 1;

    // Save the intermediate array size required for running this program
    p->intermediate_requirement = s.max_ip;
}

IntermediateRepresentation compile(const Expression& e) noexcept {
    IntermediateRepresentation p(2*e.num_of_nodes + 1);
    compile(e, &p);
    return p;
}

}