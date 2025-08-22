// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "full_initialization.hpp"
#include "initializer/full_initializer.hpp"

namespace qsr {
    std::shared_ptr<BaseInitializer> FullInitialization::get_initializer(const Config &config) {
        Config overriden_config = config;

        if (init_depth.has_value() && init_depth.value() >= 1 && init_depth.value() <= config.max_depth) {
            overriden_config.max_depth = init_depth.value();
        } else {
            overriden_config.max_depth = config.max_depth;
        }

        return std::make_shared<FullInitializer>(overriden_config);
    }
}

