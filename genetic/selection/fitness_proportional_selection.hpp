// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef FITNESS_PROPORTIONAL_SELECTION_HPP
#define FITNESS_PROPORTIONAL_SELECTION_HPP

#include "base.hpp"
#include "selector/base.hpp"

namespace qsr {

/**
 * @brief Fitness Proportional Selection strategy
 *
 * @details
 * This selection strategy selects individuals for reproduction based on their fitness.
 * Individuals with higher fitness have a higher probability of being selected.
 * This strategy is also known as roulette wheel selection.
 *
 * The calculation of fitnesses is done such that all fitness values are in the range
 * [0, +infinity]. Therefore, the selection probabilities are simply calculated by
 * dividing each fitness value by the sum of all fitness values.
 *
 */
class FitnessProportionalSelection : public BaseSelection {
public:
    std::shared_ptr<BaseSelector> get_selector(int npopulation) noexcept override;
};

}

#endif