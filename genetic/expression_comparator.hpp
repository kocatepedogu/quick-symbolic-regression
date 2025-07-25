#ifndef EXPRESSION_COMPARATOR_HPP
#define EXPRESSION_COMPARATOR_HPP

#include "../expressions/expression.hpp"

class ExpressionComparator {
public:
    /* Whether fitness(a) < fitness(b) */
    bool operator() (const Expression& a, const Expression& b);
};

#endif