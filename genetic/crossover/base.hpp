#ifndef CROSSOVER_BASE_HPP_DEFINED
#define CROSSOVER_BASE_HPP_DEFINED

#include "../../expressions/expression.hpp"

#include <tuple>

class BaseCrossover {
public:
    virtual std::tuple<Expression, Expression> crossover(Expression e1, Expression e2) noexcept;

    virtual ~BaseCrossover() = default;
};

#endif