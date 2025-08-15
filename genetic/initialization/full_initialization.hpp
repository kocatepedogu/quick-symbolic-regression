// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef INITIALIZATION_FULL_HPP
#define INITIALIZATION_FULL_HPP

#include "base.hpp"

namespace qsr {
    class FullInitialization : public BaseInitialization {
    public:
        constexpr FullInitialization(int depth) : depth(depth) {}
    
        std::shared_ptr<BaseInitializer> get_initializer(int nvars, int nweights, int npopulation) override;

    private:
        const int depth;
    };
}

#endif