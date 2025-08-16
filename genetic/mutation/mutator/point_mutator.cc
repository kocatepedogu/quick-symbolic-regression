// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "point_mutator.hpp"

#include <cassert>
#include <iostream>

#include "../../../util/rng.hpp"

namespace qsr {
    PointMutator::PointMutator(int nvars, int nweights, float mutation_probability, std::shared_ptr<FunctionSet> function_set)  :
        nvars(nvars),
        nweights(nweights),
        mutation_probability(mutation_probability),
        function_set(function_set) {}

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
                    if (function_set->sine) {
                        subtree.operation = SINE;
                        break;
                    }
                case 1:
                    if (function_set->cosine) {
                        subtree.operation = COSINE;
                        break;
                    }
                case 2:
                    if (function_set->exponential) {
                        subtree.operation = EXPONENTIAL;
                        break;
                    }
                case 3:
                    if (function_set->rectified_linear_unit) {
                        subtree.operation = RECTIFIED_LINEAR_UNIT;
                        break;
                    }
                default:
                    std::cerr << "Invalid state at line" << __LINE__ << std::endl;
                    abort();
                    break;
            }
        }

        // Check if the node is a binary operation
        if (num_operands == 2) {
            // Replace with a binary operation

            int operation = thread_local_rng() % 4;
            switch (operation) {
                case 0:
                    if (function_set->addition) {
                        subtree.operation = ADDITION;
                        break;
                    }
                case 1:
                    if (function_set->subtraction) {
                        subtree.operation = SUBTRACTION;
                        break;
                    }
                case 2:
                    if (function_set->multiplication) {
                        subtree.operation = MULTIPLICATION;
                        break;
                    }
                case 3:
                    if (function_set->division) {
                        subtree.operation = DIVISION;
                        break;
                    }
                default:
                    std::cerr << "Invalid state" << __LINE__ << std::endl;
                    abort();
                    break;
            }
        }

        return result;
    }
}