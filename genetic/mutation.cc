#include "mutation.hpp"

#include "../util/rng.hpp"
#include "../util/macro.hpp"
#include "../expressions/unary.hpp"
#include "../expressions/binary.hpp"

Expression Mutation::mutate(const Expression &expr) noexcept {
    if (((thread_local_rng() % RAND_MAX) / (float)RAND_MAX) > mutation_probability) {
        return expr;
    }

    int target = thread_local_rng() % expr.num_of_nodes;
    int current = 0;

    return mutate(expr, current, target);
}

Expression Mutation::mutate(const Expression &node, int& current, int target) noexcept {
    if (current++ == target) {
        return expression_generator.generate();
    }

    #define _MUTATE_CALL(i) \
        mutate(node.operands[i], current, target)

    switch (node.operation) {
        case CONSTANT:
        case IDENTITY:
        case PARAMETER:
            return node;

        BINARY_OP_CASE(ADDITION, _MUTATE_CALL, +);
        BINARY_OP_CASE(SUBTRACTION, _MUTATE_CALL, -);
        BINARY_OP_CASE(MULTIPLICATION, _MUTATE_CALL, *);
        BINARY_OP_CASE(DIVISION, _MUTATE_CALL, /);
        UNARY_OP_CASE(SINE, _MUTATE_CALL, Sin);
        UNARY_OP_CASE(COSINE, _MUTATE_CALL, Cos);
        UNARY_OP_CASE(EXPONENTIAL, _MUTATE_CALL, Exp);
    }

    fprintf(stderr, "Mutation::mutate Unknown operation encountered\n");
    abort();
}

