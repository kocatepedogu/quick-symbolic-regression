// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "ramped_half_and_half_initialization.hpp"
#include "initializer/ramped_half_and_half_initializer.hpp"

namespace qsr {
    std::shared_ptr<BaseInitializer> RampedHalfAndHalfInitialization::get_initializer(int nvars, int nweights, int npopulation, int max_depth, std::shared_ptr<FunctionSet> function_set) {
        int depth;

        if (init_depth.has_value() && init_depth.value() >= 1 && init_depth.value() <= max_depth) {
            depth = init_depth.value();
        } else {
            depth = max_depth;
        }

        return std::make_shared<RampedHalfAndHalfInitializer>(nvars, nweights, depth, npopulation, function_set);
    }
}
