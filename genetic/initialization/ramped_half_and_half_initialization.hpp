// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef INITIALIZATION_RAMPED_HALF_AND_HALF_HPP
#define INITIALIZATION_RAMPED_HALF_AND_HALF_HPP

#include "base.hpp"
#include <optional>

namespace qsr {
    /**
     * @brief Ramped Half-and-Half initialization strategy
     *
     * @details
     * Half of the initial population consist of individuals produced by @ref GrowInitialization,
     * while the other half consist of individuals produced by @ref FullInitialization.

     * The maximum depth is normally determined by the global configuration, which is also the highest depth
     * to which expressions can ever grow. However, this upper bound may be considered too high for 
     * initialization. Therefore, the initial maximum depth can be overriden using the `init_depth` parameter.
     *
     * @param init_depth Optional initial maximum depth for the trees. Note that this value cannot exceed
     * the maximum depth defined in the global configuration. In that case, it will be capped to the maximum depth
     * defined in the global configuration.
     */
    class RampedHalfAndHalfInitialization : public BaseInitialization {
    public:
        constexpr RampedHalfAndHalfInitialization(std::optional<int> init_depth = std::nullopt) 
            : init_depth(init_depth) {}

        std::shared_ptr<BaseInitializer> get_initializer(const Config &config) override;

    private:
        const std::optional<int> init_depth;
    };
}

#endif