#include "default.hpp"

Expression DefaultMutation::mutate(const Expression &expr) noexcept {
    const auto &random_expr = expression_generator.generate();
    const auto &result_pair = crossover.crossover(expr, random_expr);

    Expression result = 0.0;

    // Substitute random subtree into the original expression (50% chance)
    if (rand() % 2 == 0) {
        result = get<0>(result_pair);
    }
    // Substitute original expression into the random subtree (50% chance)
    else {
        result = get<1>(result_pair);
    }

    // Use the same weights as the original expression
    result.weights = expr.weights;

    // Return the mutated expression
    return result;
}

