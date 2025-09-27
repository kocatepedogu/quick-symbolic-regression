// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef MUTATOR_POINT_HPP
#define MUTATOR_POINT_HPP

#include "expressions/expression.hpp"

#include "genetic/common/expression_picker.hpp"
#include "genetic/common/expression_reorganizer.hpp"
#include "genetic/common/config.hpp"

#include "base.hpp"

namespace qsr {

class PointMutator : public BaseMutator {
public:
    PointMutator(const Config &config, double mutation_probability);

    Expression mutate(const Expression &expr) noexcept override;

private:
    const Config config;

    const double mutation_probability;

    ExpressionPicker expression_picker;

    ExpressionReorganizer expression_reorganizer;
};

}

#endif