// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef INITIALIZATION_BASE_HPP
#define INITIALIZATION_BASE_HPP

#include "initializer/base.hpp"

#include "genetic/common/config.hpp"

#include <memory>
namespace qsr {
    class BaseInitialization {
    public:
        virtual std::shared_ptr<BaseInitializer> get_initializer(const Config &config);

    protected:
        /**
         * @brief Overrides the maximum depth in the configuration
         *
         * @param config The original configuration
         * @param init_depth Optional initial maximum depth for the trees. Note that this value cannot exceed
         * the maximum depth defined in the global configuration. In that case, it will be capped to the maximum depth
         * defined in the global configuration.
         *
         * @return A new configuration with the maximum depth overridden
        */
        Config override_depth(const Config &config, const std::optional<int> &init_depth) const;
    };
}

#endif