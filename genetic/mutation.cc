#include "mutation.hpp"
#include "crossover.hpp"

Expression Mutation::mutate(const Expression &expr) noexcept {
    const auto &random_expr = expression_generator.generate();
    const auto &result_pair = crossover.crossover(expr, random_expr);

    return get<0>(result_pair);
}

