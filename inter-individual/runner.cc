#include "runner.hpp"

#include "./program/program.hpp"
#include "./vm/vm.hpp"

namespace inter_individual {
    void Runner::run(const std::vector<Expression>& population, const Dataset& dataset) {
        // Convert symbolic expression to bytecode program
        Program program_pop;
        program_create(&program_pop, population);

        // Virtual machine of the thread
        VirtualMachine *vm = new VirtualMachine(dataset);
        vm->fit(program_pop);

        // Delete machine
        delete vm;

        // Destroy programs
        program_destroy(program_pop);
    }
}
