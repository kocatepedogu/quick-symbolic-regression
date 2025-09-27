// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef MUTATOR_SUBTREE_HPP
#define MUTATOR_SUBTREE_HPP

#include "expressions/expression.hpp"

#include "genetic/common/expression_generator.hpp"
#include "genetic/common/expression_picker.hpp"
#include "genetic/common/expression_reorganizer.hpp"

#include "base.hpp"

namespace qsr {

class SubtreeMutator : public BaseMutator {
public:
    SubtreeMutator(const Config &config, double mutation_probability, int max_depth_increment);

    Expression mutate(const Expression &expr) noexcept override;

private:
    const Config config;

    const double mutation_probability;

    const int max_depth_increment;

    ExpressionGenerator expression_generator;

    ExpressionPicker expression_picker;

    ExpressionReorganizer expression_reorganizer;
};

}

#endif