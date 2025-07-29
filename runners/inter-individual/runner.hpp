#ifndef INTER_RUNNER_HPP
#define INTER_RUNNER_HPP

#include "../base.hpp"

#include "../../expressions/expression.hpp"
#include "../../dataset/dataset.hpp"
#include "./vm/vm.hpp"

#include </usr/lib/clang/20/include/omp.h>

namespace inter_individual {
    class Runner : public BaseRunner {
    private:
        std::shared_ptr<Dataset> dataset;
        const int nweights;

        VirtualMachine *vm;

    public:
        Runner(std::shared_ptr<Dataset> dataset, int nweights);

        void run(std::vector<Expression>& population, int epochs, float learning_rate) override;

        hipStream_t stream;

        ~Runner();
    };
}

#endif