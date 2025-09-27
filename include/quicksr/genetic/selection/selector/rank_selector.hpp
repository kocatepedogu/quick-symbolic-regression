// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef RANK_SELECTOR_HPP
#define RANK_SELECTOR_HPP

#include "base.hpp"

namespace qsr {

class RankSelector : public BaseSelector {
public:
    constexpr RankSelector(int npopulation, double sp) :
        npopulation(npopulation), sp(sp), indices(npopulation), probabilities(npopulation) {}

    void update(const Expression population[]) override;

    const Expression& select(const Expression population[]) override;

private:
    const int npopulation;
    const double sp; // Selection pressure parameter

    std::vector<int> indices;
    std::vector<double> probabilities;
};

}

#endif