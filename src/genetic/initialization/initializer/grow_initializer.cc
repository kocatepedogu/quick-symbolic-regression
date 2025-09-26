// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "genetic/initialization/initializer/grow_initializer.hpp"

namespace qsr {

GrowInitializer::GrowInitializer(const Config &config) :
        config(config), generator(config) {}

void GrowInitializer::initialize(std::vector<Expression>& population) {
    population.clear();
    for (int i = 0; i < config.npopulation; ++i) {
        population.push_back(generator.generate(config.max_depth, false));
    }
}

}