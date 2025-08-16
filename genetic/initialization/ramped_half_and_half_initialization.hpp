// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef INITIALIZATION_RAMPED_HALF_AND_HALF_HPP
#define INITIALIZATION_RAMPED_HALF_AND_HALF_HPP

#include "base.hpp"
#include <optional>

namespace qsr {
    class RampedHalfAndHalfInitialization : public BaseInitialization {
    public:
        constexpr RampedHalfAndHalfInitialization(std::optional<int> init_depth = std::nullopt) 
            : init_depth(init_depth) {}

        std::shared_ptr<BaseInitializer> get_initializer(int nvars, int nweights, int npopulation, int max_depth) override;

    private:
        const std::optional<int> init_depth;
    };
}

#endif