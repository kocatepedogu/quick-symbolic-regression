// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "grow_initializer.hpp"

namespace qsr {

GrowInitializer::GrowInitializer(int nvars, int nweights, int max_depth, int npopulation, std::shared_ptr<FunctionSet> function_set) :
        generator(nvars, nweights, max_depth, function_set), npopulation(npopulation) {}

void GrowInitializer::initialize(std::vector<Expression>& population) {
    population.clear();
    for (int i = 0; i < npopulation; ++i) {
        population.push_back(generator.generate());
    }
}

}