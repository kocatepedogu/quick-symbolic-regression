#include "runner_generator.hpp"
#include "runner.hpp"

namespace inter_individual {
    std::shared_ptr<BaseRunner> RunnerGenerator::generate(std::shared_ptr<Dataset> dataset, int nweights) {
        return std::make_shared<Runner>(dataset, nweights);
    }
};
