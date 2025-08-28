// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "genetic/initialization/base.hpp"

namespace qsr {
    std::shared_ptr<BaseInitializer> BaseInitialization::get_initializer(const Config &config) {
        fprintf(stderr, "Unimplemented base method called.");
        abort();
    }

    Config BaseInitialization::override_depth(const Config &config, const std::optional<int> &init_depth) const {
        Config overriden_config = config;

        if (init_depth.has_value() && init_depth.value() >= 1 && init_depth.value() <= config.max_depth) {
            // If the init_depth is provided and is within the valid range, use it
            overriden_config.max_depth = init_depth.value();
        } else {
            // Otherwise, use the maximum depth defined in the global configuration as the initialization depth
            overriden_config.max_depth = config.max_depth;
        }

        return overriden_config;
    }
}