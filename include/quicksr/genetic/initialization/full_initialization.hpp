// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef INITIALIZATION_FULL_HPP
#define INITIALIZATION_FULL_HPP

#include "genetic/initialization/base.hpp"
#include <memory>

namespace qsr {
    /**
     * @brief Full initialization strategy
     *
     * @details
     * This initialization strategy generates individuals by randomly selecting functions and terminals
     * until the maximum depth is reached in every branch of a generated tree. The resulting trees are 
     * fully grown and symmetric. All of the generated expressions have the exact depth specified by the 
     * init_depth parameter or by the global configuration if the `init_depth` parameter is not specified.
     *
     * The algorithm uses FullExpressionGenerator to construct the trees.
     *
     * @param init_depth Optional initial maximum depth for the trees. Note that this value cannot exceed
     * the maximum depth defined in the global configuration. In that case, it will be capped to the maximum depth
     * defined in the global configuration.
     */
    class FullInitialization : public BaseInitialization {
    public:
        constexpr FullInitialization(std::optional<int> init_depth = std::nullopt) 
            : init_depth(init_depth) {}

        std::shared_ptr<BaseInitializer> get_initializer(const Config &config) override;

    private:
        const std::optional<int> init_depth;
    };
}

#endif