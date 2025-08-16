// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "expression_generator.hpp"

#include "../../util/rng.hpp"
#include "../../util/macro.hpp"
#include "../../expressions/unary.hpp"
#include "../../expressions/binary.hpp"

#include <cassert>
#include <random>

namespace qsr {

ExpressionGenerator::ExpressionGenerator(int nvars, int nweights, int max_depth, std::shared_ptr<FunctionSet> function_set) : 
    nvars(nvars), nweights(nweights), max_depth(max_depth), function_set(function_set) {

    if (max_depth <= 0) {
        fprintf(stderr, "ExpressionGenerator: max_depth must be greater than zero.\n");
        abort();
    }

    if (nvars <= 0) {
        fprintf(stderr, "ExpressionGenerator: number of variables must be greater than zero.\n");
        abort();
    }

    depth_one_distribution = std::discrete_distribution<>({
        0.0,  /*CONSTANT*/
        1.0,  /*PARAMETER*/
        1.0,  /*IDENTITY*/
    });

    depth_two_distribution = std::discrete_distribution<>({
        0.0,  /*CONSTANT*/
        1.0,  /*PARAMETER*/
        1.0,  /*IDENTITY*/
        function_set->addition ? 1.0 : 0.0,
        function_set->subtraction ? 1.0 : 0.0,
        function_set->multiplication ? 1.0 : 0.0,
        function_set->division ? 1.0 : 0.0,
        function_set->sine ? 1.0 : 0.0,
        function_set->cosine ? 1.0 : 0.0,
        function_set->exponential ? 1.0 : 0.0,
        function_set->rectified_linear_unit ? 1.0 : 0.0,
    });
}

int ExpressionGenerator::random_operation(int max_depth) noexcept 
{
    // If the requested max depth is one, the only possible operations
    // are the ones that create leaf nodes: variables and weights.
    if (max_depth == 1) 
    {
        return depth_one_distribution(thread_local_rng);
    }

    // Otherwise, it is possible to add at least two to current depth.
    // Both binary and unary operations can be chosen.
    else if (max_depth >= 2) 
    {
        return depth_two_distribution(thread_local_rng);
    }

    // It is a bug if an operation with zero or negative depth is requested.
    else 
    {
        fprintf(stderr, "ExpressionGenerator::random_operation Illegal state encountered.\n");
        abort();
    }
}

Expression ExpressionGenerator::generate(int max_depth) noexcept {
    int operation = random_operation(max_depth);

    #define _RANDOM_EXPR_CALL(i) \
        generate(max_depth - 1)

    switch (operation) {
        case IDENTITY:
            return Var(thread_local_rng() % nvars);
        case PARAMETER:
            return Parameter(thread_local_rng() % nweights);

        BINARY_OP_CASE(ADDITION, _RANDOM_EXPR_CALL, +);
        BINARY_OP_CASE(SUBTRACTION, _RANDOM_EXPR_CALL, -);
        BINARY_OP_CASE(MULTIPLICATION, _RANDOM_EXPR_CALL, *);
        BINARY_OP_CASE(DIVISION, _RANDOM_EXPR_CALL, /);
        UNARY_OP_CASE(SINE, _RANDOM_EXPR_CALL, Sin);
        UNARY_OP_CASE(COSINE, _RANDOM_EXPR_CALL, Cos);
        UNARY_OP_CASE(EXPONENTIAL, _RANDOM_EXPR_CALL, Exp);
        UNARY_OP_CASE(RECTIFIED_LINEAR_UNIT, _RANDOM_EXPR_CALL, ReLU);
    }

    // Control must never reach here.
    fprintf(stderr, "ExpressionGenerator::generate Illegal state encountered.\n");
    abort();
}

Expression ExpressionGenerator::generate() noexcept {
    return generate(this->max_depth);
}

}