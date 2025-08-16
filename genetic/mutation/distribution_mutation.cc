#include "distribution_mutation.hpp"
#include "mutator/distribution_mutator.hpp"

namespace qsr {
    std::shared_ptr<BaseMutator> DistributionMutation::get_mutator(const Config &config) {
        // Create a vector to store mutators
        std::vector<std::shared_ptr<BaseMutator>> mutators;

        // Create customized mutators from given mutations
        for (const auto &mutation : mutations) {
            mutators.push_back(mutation->get_mutator(config));
        }

        // Return a DistributionMutator with the created mutators and probabilities
        return std::make_shared<DistributionMutator>(mutators, probabilities);
    }
}