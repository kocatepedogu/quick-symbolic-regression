// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "genetic/initialization/full_initialization.hpp"
#include "genetic/initialization/initializer/full_initializer.hpp"

namespace qsr {
    std::shared_ptr<BaseInitializer> FullInitialization::get_initializer(const Config &config) {
        return std::make_shared<FullInitializer>(override_depth(config, init_depth));
    }
}
