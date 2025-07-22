#include "runner.hpp"

#include "./program/program.hpp"
#include "./vm/vm.hpp"

namespace inter_individual {
    Runner::Runner(const Dataset& dataset, int nweights) :
        dataset(dataset), nweights(nweights) 
    {
        // Create virtual machine
        vm = new VirtualMachine(dataset, nweights);
    }

    void Runner::run(const std::vector<Expression>& population, int epochs, float learning_rate) {
        // Convert symbolic expression to bytecode program
        Program program_pop;
        program_create(&program_pop, population);

        // Virtual machine of the thread
        vm->fit(program_pop, epochs, learning_rate);

        // Destroy programs
        program_destroy(program_pop);
    }

    Runner::~Runner() {
        // Destroy virtual machine
        delete vm;
    }
}
