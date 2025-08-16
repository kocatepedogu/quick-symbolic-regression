// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "point_mutator.hpp"

#include <cassert>

#include "../../../util/rng.hpp"

namespace qsr {
    Expression PointMutator::mutate(const Expression &expr) noexcept {
        if (((thread_local_rng() % RAND_MAX) / (float)RAND_MAX) > mutation_probability) {
            return expr;
        }

        // Make a copy of the expression to mutate
        Expression result = expr;

        // Pick a subtree whose root node will be replaced
        auto &subtree = expression_picker.pick(result);

        // Find the number of operands of the root node
        int num_operands = subtree.operands.size();
        assert (num_operands >= 0 && num_operands <= 2);

        // Check if the node is a leaf node
        if (num_operands == 0) {
            // Replace with a parameter or variable

            int operation = thread_local_rng() % 2;
            switch (operation) {
                case 0:
                    subtree.operation = IDENTITY;
                    subtree.argindex = thread_local_rng() % nvars;
                    break;
                case 1:
                    subtree.operation = PARAMETER;
                    subtree.argindex = thread_local_rng() % nweights;
                    break;
            }
        }

        // Check if the node is a unary operation
        if (num_operands == 1) {
            // Replace with a unary operation

            int operation = thread_local_rng() % 4;
            switch (operation) {
                case 0:
                    subtree.operation = SINE;
                    break;
                case 1:
                    subtree.operation = COSINE;
                    break;
                case 2:
                    subtree.operation = EXPONENTIAL;
                    break;
                case 3:
                    subtree.operation = RECTIFIED_LINEAR_UNIT;
                    break;
            }
        }

        // Check if the node is a binary operation
        if (num_operands == 2) {
            // Replace with a binary operation

            int operation = thread_local_rng() % 4;
            switch (operation) {
                case 0:
                    subtree.operation = ADDITION;
                    break;
                case 1:
                    subtree.operation = SUBTRACTION;
                    break;
                case 2:
                    subtree.operation = MULTIPLICATION;
                    break;
                case 3:
                    subtree.operation = DIVISION;
                    break;
            }
        }

        return result;
    }
}