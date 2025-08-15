// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef RANK_SELECTION_HPP
#define RANK_SELECTION_HPP

#include "base.hpp"

namespace qsr {

class RankSelection : public BaseSelection {
public:
    constexpr RankSelection(float sp) : sp(sp) {}

    std::shared_ptr<BaseSelector> get_selector(int npopulation) noexcept override;

private:
    float sp; // Selection pressure parameter
};

}

#endif