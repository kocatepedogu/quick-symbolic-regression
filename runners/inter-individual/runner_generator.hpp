#ifndef INTERINDIVIDUAL_RUNNER_GENERATOR_HPP
#define INTERINDIVIDUAL_RUNNER_GENERATOR_HPP

#include "../runner_generator_base.hpp"

namespace inter_individual {
    class RunnerGenerator : public BaseRunnerGenerator {
    public:
        std::shared_ptr<BaseRunner> generate(std::shared_ptr<Dataset> dataset, int nweights);
    };
};

#endif