#include "runner.hpp"

#include "./program/program.hpp"
#include "./vm/vm.hpp"

#include "../util.hpp"

namespace intra_individual {
    Runner::Runner(const Dataset& dataset, const int nweights) : dataset(dataset) {
        omp_init_lock(&lock);
        omp_init_lock(&print_lock);

        #pragma omp parallel for num_threads(nstreams)
        for (int i = 0; i < nstreams; ++i) {
            // Create stream
            HIP_CALL(hipStreamCreate(&streams[i]));

            // Create virtual machine associated with the created stream
            vms[i] = new VirtualMachine(dataset, streams[i], nweights, print_lock);
        }
    }

    Runner::~Runner() {
        omp_destroy_lock(&lock);
        omp_destroy_lock(&print_lock);

        #pragma omp parallel for num_threads(nstreams)
        for (int i = 0; i < nstreams; ++i) {
            // Destroy virtual machine associated with the stream to be destroyed
            delete vms[i];

            // Destroy stream
            HIP_CALL(hipStreamDestroy(streams[i]));
        }
    }

    void Runner::run(std::vector<Expression>& population, int epochs, float learning_rate) {
        // Convert symbolic expressions to bytecode program
        Program program_pop;
        program_create(&program_pop, population);

        // Remaining work
        int work_count = population.size();

        // Parallel region to exploit remaining resources with inter-individual parallelism
        #pragma omp parallel num_threads(nstreams)
        {
            const int tid = omp_get_thread_num();

            while(true) {
                omp_set_lock(&lock);

                // Check if there is remaining work
                if (work_count == 0) {
                    // If there is no work, quit the thread.
                    omp_unset_lock(&lock);
                    break;
                } else {
                    // If there is work, fetch the bytecode program
                    const int program_idx = work_count - 1;
                    c_inst_1d code = program_pop.bytecode[program_idx];
                    const int code_length = program_pop.num_of_instructions[program_idx];

                    // Notify others about the fact that there is now less work
                    --work_count;
                    omp_unset_lock(&lock);

                    // Do the work
                    vms[tid]->fit(code, code_length, epochs, learning_rate);

                    // Compute total loss
                    float total_loss = 0;
                    #pragma omp simd reduction(+:total_loss)
                    for (int i = 0; i < dataset.m; ++i) {
                        total_loss += vms[tid]->loss_d[i];
                    }
                    population[program_idx].fitness = -total_loss;
                }
            }
        }

        // Destroy bytecode program
        program_destroy(program_pop);
    }
}
