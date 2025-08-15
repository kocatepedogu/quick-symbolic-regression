// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef INITIALIZATION_GROW_HPP
#define INITIALIZATION_GROW_HPP

#include "base.hpp"

namespace qsr {
    class GrowInitialization : public BaseInitialization {
    public:
        std::shared_ptr<BaseInitializer> get_initializer(int nvars, int nweights, int npopulation, int max_depth) override;
    };
}

#endif