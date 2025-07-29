#ifndef RUNNERS_BASE_HPP
#define RUNNERS_BASE_HPP

#include "../expressions/expression.hpp"

#include <vector>

class BaseRunner {
public:
    virtual void run(std::vector<Expression>& population, int epochs, float learning_rate);
};

#endif