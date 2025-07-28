#ifndef MUTATION_BASE_HPP
#define MUTATION_BASE_HPP

#include "../../expressions/expression.hpp"

class BaseMutation {
public:
    virtual Expression mutate(const Expression &expr) noexcept = 0;
};

#endif