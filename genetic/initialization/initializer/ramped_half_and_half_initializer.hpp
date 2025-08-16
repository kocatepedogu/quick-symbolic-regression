// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef RAMPED_HALF_AND_HALF_INITIALIZER_HPP
#define RAMPED_HALF_AND_HALF_INITIALIZER_HPP

#include "base.hpp"

#include "../../common/expression_generator.hpp"
#include "../../common/expression_generator_full.hpp"

namespace qsr {

class RampedHalfAndHalfInitializer : public BaseInitializer {
public:
    constexpr RampedHalfAndHalfInitializer(int nvars, int nweights, int max_depth, int npopulation) :
        grow_generator(nvars, nweights, max_depth), 
        full_generator(nvars, nweights, max_depth),
        npopulation(npopulation) {}

    void initialize(std::vector<Expression>& population) override;

private:
    const int npopulation;

    const ExpressionGenerator grow_generator;
    const FullExpressionGenerator full_generator;
};

}

#endif