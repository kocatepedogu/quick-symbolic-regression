// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef MUTATOR_POINT_HPP
#define MUTATOR_POINT_HPP

#include "../../../expressions/expression.hpp"

#include "../../common/expression_picker.hpp"
#include "../../common/expression_reorganizer.hpp"
#include "../../common/function_set.hpp"

#include "base.hpp"
#include <memory>

namespace qsr {

class PointMutator : public BaseMutator {
public:
    PointMutator(int nvars, int nweights, float mutation_probability, std::shared_ptr<FunctionSet> function_set);

    Expression mutate(const Expression &expr) noexcept override;

private:
    const int nvars;

    const int nweights;

    const float mutation_probability;

    ExpressionPicker expression_picker;

    ExpressionReorganizer expression_reorganizer;

    std::shared_ptr<FunctionSet> function_set;
};

}

#endif