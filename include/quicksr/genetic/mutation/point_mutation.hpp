// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef MUTATION_POINT_HPP
#define MUTATION_POINT_HPP

#include "base.hpp"

#include <memory>

namespace qsr {

/**
 * @brief Point mutation strategy
 *
 * @details
 * This mutation strategy randomly selects a node in the expression.
 *
 * If the selected node is a terminal (constant, variable, parameter), it is replaced 
 * by another randomly generated terminal.
 *
 * If the selected node is a unary operation (sin, cos, exp, relu), it is replaced by another randomly 
 * chosen unary operation, while its sole operand (child) is preserved.
 *
 * If the selected node is a binary operation (+, -, *, /), it is replaced by another randomly 
 * selected binary operation, while both of its operands (children) are preserved.
 *
 * This strategy explores new functions without growth in the complexity of the expressions.
 *
 * @param mutation_probability The probability of applying the mutation to an individual.
 */
class PointMutation : public BaseMutation {
public:
    constexpr PointMutation(float mutation_probability) :
        mutation_probability(mutation_probability) {}

    virtual std::shared_ptr<BaseMutator> get_mutator(const Config &config) override;

private:
    float mutation_probability;
};

}

#endif