// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "full_initialization.hpp"
#include "initializer/full_initializer.hpp"

namespace qsr {
    std::shared_ptr<BaseInitializer> FullInitialization::get_initializer(int nvars, int nweights, int npopulation, int max_depth) {
        return std::make_shared<FullInitializer>(nvars, nweights, max_depth, npopulation);
    }
}

