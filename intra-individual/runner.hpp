#ifndef INTRA_RUNNER_HPP
#define INTRA_RUNNER_HPP

#include "../expressions/expression.hpp"
#include "./dataset/dataset.hpp"

#include </usr/lib/clang/20/include/omp.h>

namespace intra_individual {
    class Runner {
    private:
        omp_lock_t lock;
        omp_lock_t print_lock;

    public:
        Runner();
        ~Runner();

        void run(const std::vector<Expression>& population, const Dataset& dataset);
    };
}

#endif