// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef MUTATOR_POINT_HPP
#define MUTATOR_POINT_HPP

#include "../../../expressions/expression.hpp"

#include "../../common/expression_picker.hpp"
#include "../../common/expression_reorganizer.hpp"

#include "base.hpp"

namespace qsr {

class PointMutator : public BaseMutator {
public:
    constexpr PointMutator(int nvars, int nweights, float mutation_probability)  :
        nvars(nvars),
        nweights(nweights),
        mutation_probability(mutation_probability) {}

    Expression mutate(const Expression &expr) noexcept override;

private:
    const int nvars;

    const int nweights;

    const float mutation_probability;

    ExpressionPicker expression_picker;

    ExpressionReorganizer expression_reorganizer;
};

}

#endif