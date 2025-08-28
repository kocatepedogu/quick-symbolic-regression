// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "genetic/initialization/initializer/ramped_half_and_half_initializer.hpp"

namespace qsr {

RampedHalfAndHalfInitializer::RampedHalfAndHalfInitializer(const Config &config) :
    config(config), grow_generator(config), full_generator(config) {}

    
void RampedHalfAndHalfInitializer::initialize(std::vector<Expression>& population) {
    population.clear();

    for (int i = 0; i < config.npopulation / 2; ++i) {
        population.push_back(grow_generator.generate());
    }

    for (int i = 0; i < config.npopulation / 2; ++i) {
        population.push_back(full_generator.generate());
    }
}

}