#include "crossover.hpp"

#include "../util/rng.hpp"
#include "../expressions/unary.hpp"
#include "../expressions/binary.hpp"

#include "expression_picker.hpp"

#include <tuple>

Expression reorganize(const Expression& e) {
    #define _REORGANIZE_CALL(i) \
        reorganize(e.operands[i])

    switch (e.operation) {
        case CONSTANT:
        case IDENTITY:
        case PARAMETER:
            return e;

        BINARY_OP_CASE(ADDITION, _REORGANIZE_CALL, +);
        BINARY_OP_CASE(SUBTRACTION, _REORGANIZE_CALL, -);
        BINARY_OP_CASE(MULTIPLICATION, _REORGANIZE_CALL, *);
        BINARY_OP_CASE(DIVISION, _REORGANIZE_CALL, /);
        UNARY_OP_CASE(SINE, _REORGANIZE_CALL, Sin);
        UNARY_OP_CASE(COSINE, _REORGANIZE_CALL, Cos);
        UNARY_OP_CASE(EXPONENTIAL, _REORGANIZE_CALL, Exp);
    }
}

std::tuple<Expression, Expression> Crossover::crossover(Expression e1, Expression e2) noexcept {
    if (((thread_local_rng() % RAND_MAX) / (float)RAND_MAX) > crossover_probability) {
        return std::make_tuple(e1, e2);
    }

    Expression &sub1 = expression_picker.pick(e1);
    Expression copy_sub_1(sub1);

    Expression &sub2 = expression_picker.pick(e2);
    Expression copy_sub_2(sub2);

    sub1 = copy_sub_2;
    sub2 = copy_sub_1;

    return std::make_tuple(reorganize(e1), reorganize(e2));
}