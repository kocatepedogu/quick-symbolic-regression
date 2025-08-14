// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "default.hpp"
#include "initializer/default.hpp"

namespace qsr {
    std::shared_ptr<BaseInitializer> DefaultInitialization::get_initializer(int nvars, int nweights, int npopulation) {
        return std::make_shared<DefaultInitializer>(nvars, nweights, max_depth, npopulation);
    }
}