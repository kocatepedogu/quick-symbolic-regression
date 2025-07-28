#include "default.hpp"

Expression DefaultMutation::mutate(const Expression &expr) noexcept {
    const auto &random_expr = expression_generator.generate();
    const auto &result_pair = crossover.crossover(expr, random_expr);

    // Substitute random subtree into the original expression
    if (rand() % 2 == 0) {
        return get<0>(result_pair);
    }
    // Substitute original expression into the random subtree
    else {
        return get<1>(result_pair);
    }
}

