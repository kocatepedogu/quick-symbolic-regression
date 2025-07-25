#ifndef CROSSOVER_HPP
#define CROSSOVER_HPP

#include <tuple>

#include "../expressions/expression.hpp"
#include "expression_picker.hpp"

class Crossover {
public:
    constexpr Crossover(float crossover_probability) :
        crossover_probability(crossover_probability), expression_picker() {}

    std::tuple<Expression, Expression> crossover(Expression e1, Expression e2) noexcept;

private:
    ExpressionPicker expression_picker;

    float crossover_probability;
};

#endif