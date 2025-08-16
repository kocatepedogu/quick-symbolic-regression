// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef MUTATOR_SUBTREE_HPP
#define MUTATOR_SUBTREE_HPP

#include "../../../expressions/expression.hpp"

#include "../../common/expression_generator.hpp"
#include "../../common/expression_picker.hpp"
#include "../../common/expression_reorganizer.hpp"

#include "base.hpp"

namespace qsr {

class SubtreeMutator : public BaseMutator {
public:
    SubtreeMutator(const Config &config, float mutation_probability, int max_depth_increment);

    Expression mutate(const Expression &expr) noexcept override;

private:
    const Config config;

    const float mutation_probability;

    const int max_depth_increment;

    ExpressionGenerator expression_generator;

    ExpressionPicker expression_picker;

    ExpressionReorganizer expression_reorganizer;
};

}

#endif