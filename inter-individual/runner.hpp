#ifndef INTER_RUNNER_HPP
#define INTER_RUNNER_HPP

#include "../expressions/expression.hpp"
#include "../dataset/dataset.hpp"
#include "vm/vm.hpp"

#include </usr/lib/clang/20/include/omp.h>

namespace inter_individual {
    class Runner {
    private:
        const Dataset& dataset;
        const int nweights;

        VirtualMachine *vm;

    public:
        Runner(const Dataset& dataset, int nweights);

        void run(std::vector<Expression>& population, int epochs = 500, float learning_rate = 1e-3);

        hipStream_t stream;

        ~Runner();
    };
}

#endif