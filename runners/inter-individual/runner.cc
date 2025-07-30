#include "runner.hpp"

#include "./program/program.hpp"
#include "./vm/vm.hpp"

#include "../../util/hip.hpp"
#include <hip/hip_runtime.h>

namespace inter_individual {
    Runner::Runner(std::shared_ptr<Dataset> dataset, int nweights) :
        dataset(dataset), nweights(nweights) 
    {
        // Create stream
        HIP_CALL(hipStreamCreate(&stream));

        // Create virtual machine
        vm = new VirtualMachine(dataset, nweights, stream);
    }

    void Runner::run(std::vector<Expression>& population, int epochs, float learning_rate) {
        // Convert symbolic expression to bytecode program
        Program program_pop;
        program_create(&program_pop, population);

        // Virtual machine of the thread
        auto result = vm->fit(program_pop, epochs, learning_rate);

        // Destroy programs
        program_destroy(program_pop);

        // Write loss values to the original expressions
        for (int i = 0; i < population.size(); ++i) {
            population[i].loss = result.loss_d->ptr[i];
        }

        // Write weight values to the original expressions
        for (int i = 0; i < population.size(); ++i) {
            population[i].weights.resize(nweights);
            for (int j = 0; j < nweights; ++j) {
                population[i].weights[j] = result.weights_d->ptr[j][i];
            }
        }
    }

    Runner::~Runner() {
        // Destroy virtual machine
        delete vm;

        // Destroy stream
        HIP_CALL(hipStreamDestroy(stream));
    }
}
