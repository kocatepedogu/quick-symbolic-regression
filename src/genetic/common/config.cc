#include "genetic/common/config.hpp"

#include <iostream>

namespace qsr {
    Config::Config() :
        nvars(0),
        nweights(0),
        max_depth(0),
        npopulation(0),
        function_set(nullptr) {}

    Config::Config(int nvars, int nweights, int max_depth, int npopulation, int noffspring, std::shared_ptr<FunctionSet> function_set) :
        nvars(nvars),
        nweights(nweights),
        max_depth(max_depth),
        npopulation(npopulation),
        noffspring(noffspring),
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

        if (nweights <= 0) {
            std::cerr << "Config: Number of trainable parameters (nweights) must be greater than zero." << std::endl;
            abort();
        }

        if (npopulation <= 0) {
            std::cerr << "Config: Population size (npopulation) must be greater than zero." << std::endl;
            abort();
        }
    }
}