#ifndef INTER_RUNNER_HPP
#define INTER_RUNNER_HPP

#include "../expressions/expression.hpp"
#include "./dataset/dataset.hpp"

#include </usr/lib/clang/20/include/omp.h>

namespace inter_individual {
    class Runner {
    public:
        void run(const std::vector<Expression>& population, const Dataset& dataset);
    };
}

#endif