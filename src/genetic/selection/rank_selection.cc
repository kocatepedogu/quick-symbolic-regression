// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "genetic/selection/rank_selection.hpp"
#include "genetic/selection/selector/rank_selector.hpp"

namespace qsr {
    std::shared_ptr<BaseSelector> RankSelection::get_selector(int npopulation) noexcept {
        return std::make_shared<RankSelector>(npopulation, sp);
    }
}