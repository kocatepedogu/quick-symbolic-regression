#ifndef SELECTOR_BASE_HPP
#define SELECTOR_BASE_HPP

#include "../../../expressions/expression.hpp"

class BaseSelector {
public:
    virtual void update(const Expression population[]) = 0;

    virtual const Expression& select(const Expression population[]) = 0;

    virtual ~BaseSelector();
};

#endif