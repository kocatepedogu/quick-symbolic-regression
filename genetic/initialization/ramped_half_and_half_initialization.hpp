// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef INITIALIZATION_RAMPED_HALF_AND_HALF_HPP
#define INITIALIZATION_RAMPED_HALF_AND_HALF_HPP

#include "base.hpp"

namespace qsr {
    class RampedHalfAndHalfInitialization : public BaseInitialization {
    public:
        std::shared_ptr<BaseInitializer> get_initializer(int nvars, int nweights, int npopulation, int max_depth) override;
    };
}

#endif