// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef INITIALIZATION_GROW_HPP
#define INITIALIZATION_GROW_HPP

#include "base.hpp"

namespace qsr {
    class GrowInitialization : public BaseInitialization {
    public:
        constexpr GrowInitialization(int max_depth) : max_depth(max_depth) {}
    
        std::shared_ptr<BaseInitializer> get_initializer(int nvars, int nweights, int npopulation) override;

    private:
        const int max_depth;
    };
}

#endif