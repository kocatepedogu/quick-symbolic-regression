// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "ramped_half_and_half_initialization.hpp"
#include "initializer/ramped_half_and_half_initializer.hpp"

namespace qsr {
    std::shared_ptr<BaseInitializer> RampedHalfAndHalfInitialization::get_initializer(int nvars, int nweights, int npopulation) {
        return std::make_shared<RampedHalfAndHalfInitializer>(nvars, nweights, max_depth, npopulation);
    }
}
