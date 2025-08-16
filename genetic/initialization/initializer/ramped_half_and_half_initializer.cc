// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "ramped_half_and_half_initializer.hpp"

namespace qsr {

RampedHalfAndHalfInitializer::RampedHalfAndHalfInitializer(int nvars, int nweights, int max_depth, int npopulation, std::shared_ptr<FunctionSet> function_set) :
    grow_generator(nvars, nweights, max_depth, function_set), 
    full_generator(nvars, nweights, max_depth, function_set),
    npopulation(npopulation) {}

    
void RampedHalfAndHalfInitializer::initialize(std::vector<Expression>& population) {
    population.clear();

    for (int i = 0; i < npopulation / 2; ++i) {
        population.push_back(grow_generator.generate());
    }

    for (int i = 0; i < npopulation / 2; ++i) {
        population.push_back(full_generator.generate());
    }
}

}