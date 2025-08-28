// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef RECOMBINATION_DEFAULT_HPP
#define RECOMBINATION_DEFAULT_HPP

#include "genetic/recombination/base.hpp"

namespace qsr {
    /**
     * @brief Default recombination strategy
     *
     * @details
     * A random subtree from each parent is selected and swapped to create new offspring.
     * The `crossover_probability` parameter determines the likelihood of applying the crossover to a pair of individuals.
     *
     * @param crossover_probability The probability of applying the crossover to a pair of individuals.
     */
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