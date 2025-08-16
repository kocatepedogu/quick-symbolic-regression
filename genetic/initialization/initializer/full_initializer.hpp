// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef FULL_INITIALIZER_HPP
#define FULL_INITIALIZER_HPP

#include "base.hpp"

#include "../../common/expression_generator_full.hpp"

namespace qsr {
    class FullInitializer : public BaseInitializer {
    public:
        constexpr FullInitializer(int nvars, int nweights, int depth, int npopulation) 
            : npopulation(npopulation), generator(nvars, nweights, depth) {}

        void initialize(std::vector<Expression>& population) override;

    private:
        const int npopulation;

        const FullExpressionGenerator generator;
    };
}

#endif