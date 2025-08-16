// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef GROW_INITIALIZER_HPP
#define GROW_INITIALIZER_HPP

#include "base.hpp"

#include "../../common/expression_generator.hpp"

namespace qsr {

class GrowInitializer : public BaseInitializer {
public:
    constexpr GrowInitializer(int nvars, int nweights, int max_depth, int npopulation) :
        generator(nvars, nweights, max_depth), npopulation(npopulation) {}

    void initialize(std::vector<Expression>& population) override;

private:
    const int npopulation;

    const ExpressionGenerator generator;
};

}

#endif