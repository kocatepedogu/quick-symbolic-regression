#include "crossover.hpp"

#include "../util/rng.hpp"
#include "expression_picker.hpp"

#include <tuple>

std::tuple<Expression, Expression> Crossover::crossover(Expression e1, Expression e2) noexcept {
    if (((thread_local_rng() % RAND_MAX) / (float)RAND_MAX) > crossover_probability) {
        return std::make_tuple(e1, e2);
    }

    Expression &sub1 = expression_picker.pick(e1);
    Expression &sub2 = expression_picker.pick(e2);

    Expression copy_sub_1(sub1);
    Expression copy_sub_2(sub2);

    sub1 = copy_sub_2;
    sub2 = copy_sub_1;

    return std::make_tuple(e1, e2);
}