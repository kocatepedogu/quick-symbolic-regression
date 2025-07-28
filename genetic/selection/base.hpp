#ifndef SELECTION_BASE_HPP
#define SELECTION_BASE_HPP

#include "../../expressions/expression.hpp"

class BaseSelection {
public:
    virtual void initialize(const Expression population[]) = 0;

    virtual const Expression& select(const Expression population[]) = 0;
};

#endif