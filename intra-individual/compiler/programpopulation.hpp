#ifndef INTRA_PROGRAMPOPULATION_HPP
#define INTRA_PROGRAMPOPULATION_HPP

#include "../../compiler/ir.hpp"
#include "../../expressions/expression.hpp"

#include <vector>

namespace intra_individual {
    struct ProgramPopulation {
        /// Programs
        Instruction **bytecode;

        /// Length of Programs
        int *num_of_instructions;

        /// Total number of individuals
        int num_of_individuals;
    };

    void program_create(ProgramPopulation *prog_pop, const std::vector<Expression>& exp_pop);

    void program_destroy(ProgramPopulation &prog_pop);
}

#endif