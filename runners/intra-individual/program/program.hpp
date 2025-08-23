// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef INTRA_PROGRAM_HPP
#define INTRA_PROGRAM_HPP

#include "../../../compiler/ir.hpp"
#include "../../../expressions/expression.hpp"

#include "../../../util/arrays/array1d.hpp"
#include "../../../util/arrays/array2d.hpp"

#include <vector>

namespace qsr::intra_individual {
    struct ProgramIterator;

    /**
     * @brief Stores bytecode instructions of a population of programs for intra-individual GPU execution.
     *
     * @details
     * Although the intra-individual mode executes each expression one-by-one in a sequential manner,
     * it is still beneficial (up to 4x speedup) to compile all expressions at once and then store all 
     * programs in a single contiguous block of memory due to cache behavior.
     * 
     * This class stores instructions in a 2D array (continuous in memory) with dimensions
     * [num_of_individuals, max_num_of_instructions]. The other information is stored in 1D arrays.
     *
     */
    class Program {
    public:
        /// Programs
        Array2D<Instruction> bytecode;

        /// Length of Programs
        Array1D<int> num_of_instructions;

        /// Stack requirement of Programs
        Array1D<int> stack_req;

        /// Intermediate requirement of Programs
        Array1D<int> intermediate_req;

        /// Total number of individuals
        int num_of_individuals;

        /// Length of the longest program
        int max_num_of_instructions;

        /// Constructor
        Program(const std::vector<Expression>& exp_pop);

        /// Default constructor
        Program() = default;

        /// Begin iterator for accessing individual programs
        ProgramIterator begin();

        /// End iterator for accessing individual programs
        ProgramIterator end();

    private:
        void allocate_arrays(const int num_of_individuals);

        void compile_expressions(const std::vector<Expression>& exp_pop);

        void find_longest_ir();

        void copy_to_gpu_memory();

        std::vector<IntermediateRepresentation> irs;
    };

    /**
     * @brief Represents an individual program
     *
     * @details
     * While the `Program` class stores information about programs in a structure-of-arrays 
     * form, this struct provides an array-of-structures view of the same
     * data for convenience.
     *
     * @note
     * This struct is intended to be used with iterators
     */
    struct ProgramIndividual {
        Ptr1D<Instruction> bytecode;

        int num_of_instructions;

        int stack_req;

        int intermediate_req;

        int index;
    };

    /**
     * @brief Iterator for accessing individual programs in a `Program` object.
     *
     * @details
     * This iterator allows for easy access to individual programs in a `Program` object.
     * It provides an array-of-structures view of the same data for convenience.
     *
     * @note
     * This iterator is intended to be used with the `Program` class and should not be used directly
     */
    struct ProgramIterator {
        inline ProgramIterator(Program *program) : 
            program(program), index(0) {}

        inline ProgramIterator(Program *program, int index) : 
            program(program), index(index) {}

        inline const ProgramIndividual operator*() {
            return ProgramIndividual{
                .bytecode = program->bytecode.ptr(index),
                .num_of_instructions = program->num_of_instructions.ptr[index],
                .stack_req = program->stack_req.ptr[index],
                .intermediate_req = program->intermediate_req.ptr[index],
                .index = index
            };
        }

        inline ProgramIterator &operator++() {
            ++index;
            return *this;
        }

        inline bool operator!=(ProgramIterator &rhs) {
            return index != rhs.index;
        }

    private:
        Program *program;

        int index;
    };
}

#endif