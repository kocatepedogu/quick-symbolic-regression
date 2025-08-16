// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "full_initializer.hpp"

namespace qsr {
    FullInitializer::FullInitializer(const Config &config) 
            : config(config), generator(config) {}

    void FullInitializer::initialize(std::vector<Expression> &population) {
        population.clear();
        for (int i = 0; i < config.npopulation; ++i) {
            population.push_back(generator.generate());
        }
    }
}
