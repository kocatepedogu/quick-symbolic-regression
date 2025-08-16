// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "full_initializer.hpp"

namespace qsr {
    FullInitializer::FullInitializer(int nvars, int nweights, int depth, int npopulation, std::shared_ptr<FunctionSet> function_set) 
            : npopulation(npopulation), generator(nvars, nweights, depth, function_set) {}

    void FullInitializer::initialize(std::vector<Expression>& population) {
        population.clear();
        for (int i = 0; i < npopulation; ++i) {
            population.push_back(generator.generate());
        }
    }
}
