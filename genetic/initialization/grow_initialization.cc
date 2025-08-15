// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "grow_initialization.hpp"
#include "initializer/grow_initializer.hpp"

namespace qsr {
    std::shared_ptr<BaseInitializer> GrowInitialization::get_initializer(int nvars, int nweights, int npopulation) {
        return std::make_shared<GrowInitializer>(nvars, nweights, max_depth, npopulation);
    }
}