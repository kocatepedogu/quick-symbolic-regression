#ifndef RUNNER_GENERATOR_BASE_HPP
#define RUNNER_GENERATOR_BASE_HPP

#include "base.hpp"
#include "../dataset/dataset.hpp"

#include <memory>

class BaseRunnerGenerator {
public:
    virtual std::shared_ptr<BaseRunner> generate(std::shared_ptr<Dataset> dataset, int nweights);
};

#endif