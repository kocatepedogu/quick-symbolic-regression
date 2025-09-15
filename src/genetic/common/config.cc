#include "genetic/common/config.hpp"

#include <iostream>

namespace qsr {
    Config::Config() :
        nvars(0),
        nweights(0),
        max_depth(0),
        npopulation(0),
        elite_rate(0.0f),
        survival_rate(0.0f),
        migration_rate(0.0f),
        function_set(nullptr) {}

    Config::Config(int nvars, int nweights, int max_depth, int npopulation,
                   float elite_rate, float survival_rate, float migration_rate,
                   std::shared_ptr<FunctionSet> function_set) :
        nvars(nvars),
        nweights(nweights),
        max_depth(max_depth),
        npopulation(npopulation),
        elite_rate(elite_rate),
        survival_rate(survival_rate),
        migration_rate(migration_rate),
        function_set(function_set)
    {
        if (max_depth <= 0) {
            std::cerr << "Config: Depth limit of expressions (max_depth) must be greater than zero." << std::endl;
            abort();
        }

        if (nvars <= 0) {
            std::cerr << "Config: Number of variables (nvars) must be greater than zero." << std::endl;
            abort();
        }

        if (npopulation <= 0) {
            std::cerr << "Config: Population size (npopulation) must be greater than zero." << std::endl;
            abort();
        }
    }
}