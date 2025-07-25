#ifndef EXPRESSION_PICKER_HPP
#define EXPRESSION_PICKER_HPP

#include "../expressions/expression.hpp"

class ExpressionPicker {
public:
    Expression& pick(Expression &expr) noexcept;

private:
    void pick(Expression *node, Expression **result, int& current, int target) noexcept;
};

#endif