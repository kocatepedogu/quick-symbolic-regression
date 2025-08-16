#include "distribution_mutation.hpp"
#include "mutator/distribution_mutator.hpp"

namespace qsr {
    std::shared_ptr<BaseMutator> DistributionMutation::get_mutator(int nvars, int nweights, int max_depth, std::shared_ptr<FunctionSet> function_set) {
        // Create a vector to store mutators
        std::vector<std::shared_ptr<BaseMutator>> mutators;

        // Create customized mutators from given mutations
        for (const auto &mutation : mutations) {
            mutators.push_back(mutation->get_mutator(nvars, nweights, max_depth, function_set));
        }

        // Return a DistributionMutator with the created mutators and probabilities
        return std::make_shared<DistributionMutator>(mutators, probabilities);
    }
}