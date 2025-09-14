#include "genetic/common/expression_generator_full.hpp"

#include "util/rng.hpp"
#include "util/macro.hpp"
#include "expressions/unary.hpp"
#include "expressions/binary.hpp"
#include "genetic/common/config.hpp"

#include <cassert>

namespace qsr {

FullExpressionGenerator::FullExpressionGenerator(const Config &config) : config(config) {
    depth_one_distribution = std::discrete_distribution<>({
        1.0,                                   /*CONSTANT*/
        static_cast<double>(config.nweights),  /*PARAMETER*/
        static_cast<double>(config.nvars)   ,  /*IDENTITY*/
    });

    depth_two_distribution = std::discrete_distribution<>({
        0.0,  /*CONSTANT*/
        0.0,  /*PARAMETER*/
        0.0,  /*IDENTITY*/
        config.function_set->addition ? 1.0 : 0.0,
        config.function_set->subtraction ? 1.0 : 0.0,
        config.function_set->multiplication ? 1.0 : 0.0,
        config.function_set->division ? 1.0 : 0.0,
        config.function_set->sine ? 1.0 : 0.0,
        config.function_set->cosine ? 1.0 : 0.0,
        config.function_set->exponential ? 1.0 : 0.0,
        config.function_set->rectified_linear_unit ? 1.0 : 0.0,
    });
}

Expression FullExpressionGenerator::generate(int remaining_depth) noexcept {
    #define _RANDOM_EXPR_CALL(i) \
        generate(remaining_depth - 1)

    assert(remaining_depth > 0);

    if (remaining_depth == 1) {
        int operation = depth_one_distribution(thread_local_rng);
        switch (operation) {
            case CONSTANT:
                return 2 * (thread_local_rng() % RAND_MAX) / (double)RAND_MAX - 1;
            case IDENTITY:
                return Var(thread_local_rng() % config.nvars);
            case PARAMETER:
                return Parameter(thread_local_rng() % config.nweights);
        }
    } 
    else if (remaining_depth >= 2) {
        int operation = depth_two_distribution(thread_local_rng);
        switch (operation) {
            BINARY_OP_CASE(ADDITION, _RANDOM_EXPR_CALL, +);
            BINARY_OP_CASE(SUBTRACTION, _RANDOM_EXPR_CALL, -);
            BINARY_OP_CASE(MULTIPLICATION, _RANDOM_EXPR_CALL, *);
            BINARY_OP_CASE(DIVISION, _RANDOM_EXPR_CALL, /);
            UNARY_OP_CASE(SINE, _RANDOM_EXPR_CALL, Sin);
            UNARY_OP_CASE(COSINE, _RANDOM_EXPR_CALL, Cos);
            UNARY_OP_CASE(EXPONENTIAL, _RANDOM_EXPR_CALL, Exp);
            UNARY_OP_CASE(RECTIFIED_LINEAR_UNIT, _RANDOM_EXPR_CALL, ReLU);
        }
    }

    // Control must never reach here.
    fprintf(stderr, "ExpressionGenerator::generate Illegal state encountered.\n");
    abort();
}

Expression FullExpressionGenerator::generate() noexcept {
    return generate(config.max_depth);
}

}