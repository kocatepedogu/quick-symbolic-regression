// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef MUTATOR_DEFAULT_HPP
#define MUTATOR_DEFAULT_HPP

#include "../../../expressions/expression.hpp"
#include "../../recombiner/default.hpp"
#include "../../expression_generator.hpp"
#include "base.hpp"

namespace qsr {

class DefaultMutator : public BaseMutator {
public:
    constexpr DefaultMutator(int nvars, int nweights, int max_depth, float mutation_probability)  :
        expression_generator(nvars, nweights, max_depth), 
        recombiner(mutation_probability) {}

    Expression mutate(const Expression &expr) noexcept;

private:
    ExpressionGenerator expression_generator;

    DefaultRecombiner recombiner;
};

}

#endif