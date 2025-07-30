#ifndef INTER_RUNNER_HPP
#define INTER_RUNNER_HPP

#include "../base.hpp"

#include "./program/program.hpp"

#include "../../expressions/expression.hpp"
#include "../../dataset/dataset.hpp"

#include "../../../util/hip.hpp"

#include </usr/lib/clang/20/include/omp.h>

namespace inter_individual {
    struct VirtualMachineResult {
        std::unique_ptr<Array2D<float>> weights_d;
        std::unique_ptr<Array1D<float>> loss_d;
    };

    class Runner : public BaseRunner {
    private:
        std::shared_ptr<Dataset> dataset;

        const int nweights;

        HIPState config;

    public:
        Runner(std::shared_ptr<Dataset> dataset, int nweights);

        VirtualMachineResult run(const Program &program, int epochs, float learning_rate);

        void run(std::vector<Expression>& population, int epochs, float learning_rate) override;
    };
}

#endif