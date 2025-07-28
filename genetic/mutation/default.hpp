#ifndef MUTATION_DEFAULT_HPP
#define MUTATION_DEFAULT_HPP

#include "../../expressions/expression.hpp"
#include "../crossover/default.hpp"
#include "../expression_generator.hpp"

class DefaultMutation {
public:
    constexpr DefaultMutation(int nvars, int nweights, int max_depth, float mutation_probability)  :
        expression_generator(nvars, nweights, max_depth), 
        crossover(mutation_probability) {}

    Expression mutate(const Expression &expr) noexcept;

private:
    ExpressionGenerator expression_generator;

    DefaultCrossover crossover;
};

#endif