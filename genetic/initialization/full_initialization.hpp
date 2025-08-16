// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef INITIALIZATION_FULL_HPP
#define INITIALIZATION_FULL_HPP

#include "base.hpp"
#include <memory>

namespace qsr {
    class FullInitialization : public BaseInitialization {
    public:
        constexpr FullInitialization(std::optional<int> init_depth = std::nullopt) 
            : init_depth(init_depth) {}

        std::shared_ptr<BaseInitializer> get_initializer(int nvars, int nweights, int npopulation, int max_depth, std::shared_ptr<FunctionSet> function_set) override;

    private:
        const std::optional<int> init_depth;
    };
}

#endif