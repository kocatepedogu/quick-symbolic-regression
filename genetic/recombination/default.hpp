// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef RECOMBINATION_DEFAULT_HPP
#define RECOMBINATION_DEFAULT_HPP

#include "base.hpp"

namespace qsr {
    class DefaultRecombination : public BaseRecombination {
    public:
        constexpr DefaultRecombination(float crossover_probability) : 
            crossover_probability(crossover_probability) {}

        std::shared_ptr<BaseRecombiner> get_recombiner(int max_depth) override;

    private:
        const float crossover_probability;
    };
}

#endif