#ifndef MUTATION_HPP
#define MUTATION_HPP

#include "../expressions/expression.hpp"
#include "expression_generator.hpp"

class Mutation {
public:
    constexpr Mutation(int nvars, int nweights, int max_depth, float mutation_probability)  :
        expression_generator(nvars, nweights, max_depth), 
        mutation_probability(mutation_probability) {}

    Expression mutate(const Expression &expr) noexcept;

private:
    Expression mutate(const Expression &node, int& current, int target) noexcept;

    ExpressionGenerator expression_generator;

    const float mutation_probability;
};

#endif