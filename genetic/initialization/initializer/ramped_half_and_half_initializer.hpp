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
    RampedHalfAndHalfInitializer(int nvars, int nweights, int max_depth, int npopulation, std::shared_ptr<FunctionSet> function_set);

    void initialize(std::vector<Expression>& population) override;

private:
    const int npopulation;

    ExpressionGenerator grow_generator;
    FullExpressionGenerator full_generator;
};

}

#endif