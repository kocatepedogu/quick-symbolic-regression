// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef INITIALIZATION_GROW_HPP
#define INITIALIZATION_GROW_HPP

#include "genetic/initialization/base.hpp"
#include <optional>

namespace qsr {
    /**
     * @brief Grow initialization strategy
     *
     * @details
     * This initialization strategy generates individuals by randomly selecting terminals and functions
     * until the maximum depth is reached or until a terminal is chosen. The resulting trees have varying 
     * depth and number of nodes since some branches of the trees may terminate earlier than others. 
     *
     * The smallest depth is one, which means that a generated tree can consist of a single terminal.
     * The maximum depth is normally determined by the global configuration, which is also the highest depth
     * to which expressions can ever grow. However, this upper bound may be considered too high for 
     * initialization. Therefore, the initial maximum depth can be overriden using the `init_depth` parameter.
     *
     * The algorithm uses ExpressionGenerator to construct the trees.
     *
     * @param init_depth Optional initial maximum depth for the trees. Note that this value cannot exceed
     * the maximum depth defined in the global configuration. In that case, it will be capped to the maximum depth
     * defined in the global configuration.
     */

    class GrowInitialization : public BaseInitialization {
    public:
        constexpr GrowInitialization(std::optional<int> init_depth = std::nullopt) 
            : init_depth(init_depth) {}

        std::shared_ptr<BaseInitializer> get_initializer(const Config &config) override;

    private:
        const std::optional<int> init_depth;
    };
}

#endif