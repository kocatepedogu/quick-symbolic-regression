#include "runner.hpp"

#include "./compiler/programpopulation.hpp"
#include "./vm/vm.hpp"

#include "../util.hpp"

namespace intra_individual {
    Runner::Runner() {
        omp_init_lock(&lock);
        omp_init_lock(&print_lock);
    }

    Runner::~Runner() {
        omp_destroy_lock(&lock);
        omp_destroy_lock(&print_lock);
    }

    void Runner::run(const std::vector<Expression>& population, const Dataset& dataset) {
        // Convert symbolic expression to bytecode program
        ProgramPopulation program_pop;
        program_create(&program_pop, population);

        // Number of streams
        constexpr int nstreams = 2;

        // Remaining work
        int work_count = population.size();

        // Parallel region
        #pragma omp parallel num_threads(nstreams)
        {
            // Stream of the thread
            hipStream_t stream;
            HIP_CALL(hipStreamCreate(&stream));

            // Virtual machine of the thread
            VirtualMachine *vm = new VirtualMachine(dataset, stream, 2, print_lock);

            // Fit
            while(true) {
                omp_set_lock(&lock);
                if (work_count == 0) {
                    omp_unset_lock(&lock);
                    break;
                } else {
                    const Instruction *code = program_pop.bytecode[work_count - 1];
                    const int code_length = program_pop.num_of_instructions[work_count - 1];

                    --work_count;
                    omp_unset_lock(&lock);

                    vm->fit(code, code_length);
                }
            }

            // Delete machine
            delete vm;

            // Delete stream
            HIP_CALL(hipStreamDestroy(stream));
        }

        // Destroy programs
        program_destroy(program_pop);
    }
}
