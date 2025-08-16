// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef MUTATOR_HOIST_HPP
#define MUTATOR_HOIST_HPP

#include "../../../expressions/expression.hpp"

#include "../../common/expression_picker.hpp"
#include "../../common/expression_reorganizer.hpp"

#include "base.hpp"

namespace qsr {

class HoistMutator : public BaseMutator {
public:
    constexpr HoistMutator(float mutation_probability)  :
        mutation_probability(mutation_probability) {}

    Expression mutate(const Expression &expr) noexcept override;

private:
    const float mutation_probability;

    ExpressionPicker expression_picker;

    ExpressionReorganizer expression_reorganizer;
};

}

#endif