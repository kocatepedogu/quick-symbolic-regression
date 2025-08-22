// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "grow_initialization.hpp"
#include "initializer/grow_initializer.hpp"

namespace qsr {
    std::shared_ptr<BaseInitializer> GrowInitialization::get_initializer(const Config &config) {
        return std::make_shared<GrowInitializer>(override_depth(config, init_depth));
    }
}