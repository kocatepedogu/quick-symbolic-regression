// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "genetic/recombination/default.hpp"
#include "genetic/recombination/recombiner/default.hpp"

namespace qsr {
    std::shared_ptr<BaseRecombiner> DefaultRecombination::get_recombiner(int max_depth) {
        return std::make_shared<DefaultRecombiner>(max_depth, crossover_probability);
    }
}