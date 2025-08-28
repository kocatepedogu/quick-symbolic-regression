// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef EXPRESSION_REORGANIZER_HPP
#define EXPRESSION_REORGANIZER_HPP

#include "expressions/expression.hpp"

namespace qsr {

/**
 * @brief Restores algorithmic invariants and performs algebraic optimizations
 */
class ExpressionReorganizer {
public:
    /**
     * @brief Restores algorithmic invariants and performs algebraic optimizations
     *
     * @details
     * Each expression has two values crucial for algorithmic correctness:
     * (1) `num_of_nodes` and (2) `depth`. Both of these are calculated by the constructor(s) of the
     * Expression class, and any code that utilize the constructors for producing new expressions
     * (e.g., `a + b`) preserve these invariants.
     *
     * The genetic operators, however, modify expression instances directly, leading to corrupted
     * invariants. In the subtree mutation, for example, a randomly chosen subtree of the original
     * tree is replaced by a randomly generated subtree, simply by assignment.
     * This change requires `num_of_nodes` and `depth values` of all parent nodes in the resulting tree
     * to be updated, which is not performed in a simple assignment.
     *
     * This method is called after such a corrupting operation to fix the resulting expression.
     * It works by recursively traversing the tree and regenerating the same expression
     * using the constructors at every node. This also allows omitted algebraic optimizations 
     * to be performed (e.g., addition of two constants should be simplified to a single constant).
     */
    Expression reorganize(const Expression &expr) const noexcept;
};

} // namespace qsr

#endif // EXPRESSION_REORGANIZER_HPP