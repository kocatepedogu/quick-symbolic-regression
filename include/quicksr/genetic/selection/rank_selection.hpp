// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef RANK_SELECTION_HPP
#define RANK_SELECTION_HPP

#include "genetic/selection/base.hpp"

namespace qsr {

/**
 * @brief Rank Selection strategy
 *
 * @details
 * This selection strategy selects individuals for reproduction based on their rank.
 * Individuals are first sorted based on their fitness, and the selection probability is
 * determined by their order in the sorted array.
 *
 * The calculation of selection probabilities is done by the formula
 * \[
 * P(i) = \frac{1}{n} \left(1 + sp \left(1 - \frac{2(i-1)}{n-1}\right)\right)
 * \]
 * where \( i \) is the rank of the individual, \( n \) is the population size, and \( sp \) is the selection pressure.
 *
 * The `sp` parameter (selection pressure) controls the steepness of the selection curve.
 *
 */
class RankSelection : public BaseSelection {
public:
    constexpr RankSelection(double sp) : sp(sp) {}

    std::shared_ptr<BaseSelector> get_selector(int npopulation) noexcept override;

private:
    double sp; // Selection pressure parameter
};

}

#endif