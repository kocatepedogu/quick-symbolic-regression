#include "runner.hpp"

#include "./program/program.hpp"
#include "./vm/vm.hpp"

#include "../util.hpp"

namespace inter_individual {
    Runner::Runner(const Dataset& dataset, int nweights) :
        dataset(dataset), nweights(nweights) 
    {
        // Create virtual machine
        vm = new VirtualMachine(dataset, nweights);
    }

    void Runner::run(std::vector<Expression>& population, int epochs, float learning_rate) {
        // Convert symbolic expression to bytecode program
        Program program_pop;
        program_create(&program_pop, population);

        // Create array to store loss of each function
        float *loss_d;
        init_arr_1d(loss_d, population.size());

        // Virtual machine of the thread
        vm->fit(program_pop, loss_d, epochs, learning_rate);

        // Destroy programs
        program_destroy(program_pop);

        // Write fitness values (negative loss) to the original expressions
        for (int i = 0; i < population.size(); ++i) {
            population[i].fitness = -loss_d[i];
        }

        // Destroy loss array
        del_arr_1d(loss_d);
    }

    Runner::~Runner() {
        // Destroy virtual machine
        delete vm;
    }
}
