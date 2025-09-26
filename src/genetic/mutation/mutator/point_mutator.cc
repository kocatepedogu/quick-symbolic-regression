// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "genetic/mutation/mutator/point_mutator.hpp"

#include <cassert>
#include <iostream>

#include "util/rng.hpp"
#include "util/precision.hpp"

namespace qsr {
    PointMutator::PointMutator(const Config &config, double mutation_probability)  :
        config(config), mutation_probability(mutation_probability) {}


    Expression PointMutator::mutate(const Expression &expr) noexcept {
        if (((thread_local_rng() % RAND_MAX) / (double)RAND_MAX) > mutation_probability) {
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

            switch (int operation = thread_local_rng() % 2) {
                case 0:
                    if (config.nweights != 0) {
                        subtree.operation = PARAMETER;
                        subtree.argindex = thread_local_rng() % config.nweights;
                        break;
                    }
                case 1:
                    subtree.operation = IDENTITY;
                    subtree.argindex = thread_local_rng() % config.nvars;
                    break;
            }
        }

        // Check if the node is a unary operation
        if (num_operands == 1) {
            // Replace with a unary operation

            int operation = thread_local_rng() % 4;
            switch (operation) {
                case 0:
                    if (config.function_set->sine) {
                        subtree.operation = SINE;
                        break;
                    }
                case 1:
                    if (config.function_set->cosine) {
                        subtree.operation = COSINE;
                        break;
                    }
                case 2:
                    if (config.function_set->exponential) {
                        subtree.operation = EXPONENTIAL;
                        break;
                    }
                case 3:
                    if (config.function_set->rectified_linear_unit) {
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
                    if (config.function_set->addition) {
                        subtree.operation = ADDITION;
                        break;
                    }
                case 1:
                    if (config.function_set->subtraction) {
                        subtree.operation = SUBTRACTION;
                        break;
                    }
                case 2:
                    if (config.function_set->multiplication) {
                        subtree.operation = MULTIPLICATION;
                        break;
                    }
                case 3:
                    if (config.function_set->division) {
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