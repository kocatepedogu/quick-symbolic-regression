// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef INITIALIZATION_GROW_HPP
#define INITIALIZATION_GROW_HPP

#include "base.hpp"
#include <optional>

namespace qsr {
    class GrowInitialization : public BaseInitialization {
    public:
        constexpr GrowInitialization(std::optional<int> init_depth = std::nullopt) 
            : init_depth(init_depth) {}

        std::shared_ptr<BaseInitializer> get_initializer(int nvars, int nweights, int npopulation, int max_depth, std::shared_ptr<FunctionSet> function_set) override;

    private:
        const std::optional<int> init_depth;
    };
}

#endif