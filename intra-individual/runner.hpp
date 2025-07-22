#ifndef INTRA_RUNNER_HPP
#define INTRA_RUNNER_HPP

#include "../expressions/expression.hpp"
#include "../dataset/dataset.hpp"
#include "vm/vm.hpp"

#include </usr/lib/clang/20/include/omp.h>

namespace intra_individual {
    class Runner {
    private:
        // Number of streams
        static constexpr int nstreams = 2;

        omp_lock_t lock;
        omp_lock_t print_lock;

        const Dataset& dataset;

        hipStream_t streams[nstreams];
        VirtualMachine *vms[nstreams];

    public:
        Runner(const Dataset& dataset, const int nweights);

        void run(const std::vector<Expression>& population, int epochs = 500, float learning_rate = 1e-3);

        ~Runner();
    };
}

#endif