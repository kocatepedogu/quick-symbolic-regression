#include "runner_generator_base.hpp"

#include <cstdio>

std::shared_ptr<BaseRunner> BaseRunnerGenerator::generate(std::shared_ptr<Dataset> dataset, int nweights) {
    fprintf(stderr, "Unimplemented base method called.");
    abort();
}