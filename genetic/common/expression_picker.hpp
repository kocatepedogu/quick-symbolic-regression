// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef EXPRESSION_PICKER_HPP
#define EXPRESSION_PICKER_HPP

#include "../../expressions/expression.hpp"

namespace qsr {

/**
 * @brief Extracts random subtrees from given expressions
 */
class ExpressionPicker {
public:
    /**
     * @brief Returns a reference to a randomly selected subtree within the given tree
     */
    Expression& pick(Expression &expr) noexcept;

private:
    /**
     * @brief Performs a pre-order traversal on the expression's tree to choose a subtree
     *
     * @param node The node at which the traversal starts
     * @param result The result variable to which the address of the chosen subtree's root node will be written
     * @param current The pre-order index of the node the algorithm is currently traversing
     * @param target The pre-order index of the chosen subtree's root node
     *
     * @details
     * Every expression has a `num_of_nodes` value, which is computed during the construction of expressions.
     * Terminals (constants, variables, weights) have a num_of_nodes value of one.
     * Others have a num_of_nodes value equal to the sum of nodes in their operand(s), plus one.
     *
     * The `pick` method declared above (the public one) generates a random integer in the range `[0, num_of_nodes - 1]`.
     * The chosen integer is substituted into the `target` argument of this method, which traverses
     * the tree until the pre-order index of the current node (the parameter `current`) equals `target`.
     *
     * This is to ensure that a subtree can be selected with no bias with respect to the
     * overall tree structure.
     */
    void pick(Expression *node, Expression **result, int& current, int target) noexcept;
};

}

#endif
