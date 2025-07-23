// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef EXPRESSION_H
#define EXPRESSION_H

#include <ostream>
#include <vector>

#define BINARY_OP_CASE(case_name, operand, operator_token) \
    case case_name: { \
        auto r1 = operand(0); \
        auto r2 = operand(1); \
        return r1 operator_token r2; \
    }

#define UNARY_OP_CASE(case_name, operand, operator_token) \
    case case_name: return operator_token(operand(0))

typedef enum {
    CONSTANT = 0,
    PARAMETER = 1,
    IDENTITY = 2,
    ADDITION = 3,
    SUBTRACTION = 4,
    MULTIPLICATION = 5,
    DIVISION = 6,
    SINE = 7,
    COSINE = 8,
    EXPONENTIAL = 9
} operation_t;

struct Expression
{
    constexpr Expression() noexcept : 
        Expression(0.0) {}

    /**
      * @brief Constructs a constant expression
      * @code{.cpp}
      * Expression f = 3;
      * @endcode
      */ 
    constexpr Expression(float value) noexcept :
        operation(CONSTANT), 
        value(value), 
        num_of_nodes(1) {}

    /**
      * @brief Constructs an expression that corresponds to a variable (input feature) or trainable parameter (weight)
      * @code{.cpp}
      * Expression f = Var(0);
      * Expression g = Parameter(0);
      * @endcode
      */
    constexpr Expression(operation_t operation, int argindex) noexcept :
        operation(operation), 
        argindex(argindex), 
        num_of_nodes(1) {}

    /**
      * @brief Constructs an expression in which a unary function is applied to the result of another expression.
      * @code{.cpp}
      * Expression f = Sin(e1);
      * Expression g = Exp(e2);
      * @endcode
      */ 
    Expression(operation_t operation, const Expression& e) noexcept;

    /**
      * @brief Constructs an expression in which a binary operator is applied to two expressions.
      * @code{.cpp}
      * Expression f = e1 + e2;
      * @endcode
      */ 
    Expression(operation_t operation, const Expression& e1, const Expression& e2) noexcept;

    /// Operation type of the expression
    operation_t operation;

    /// For constants
    float value;

    /// For variables and trainable parameters
    int argindex;

    /// For unary operations, binary operations and conditionals
    std::vector<Expression> operands;

    /// Total number of nodes in the tree representing the expression
    int num_of_nodes;

    /// Training loss (MSE on dataset)
    float loss;
};

/**
  * @brief Yields infix representation of the expression
  */
std::ostream& operator << (std::ostream& os, const Expression& m) noexcept;

/**
  * @brief Tests whether two expressions are the same
  */
bool operator == (const Expression& left_operand, const Expression& right_operand) noexcept;

/**
  * @brief Tests whether two expressions are different
  */
bool operator != (const Expression& left_operand, const Expression& right_operand) noexcept;

/**
  * @brief Generates a variable associated with a particular input feature
  */
static inline Expression Var(int argindex) { 
    return Expression(IDENTITY, argindex); 
}

/**
  * @brief Generates a parameter associated with a particular trainable weight
  */
static inline Expression Parameter(int argindex) {  
    return Expression(PARAMETER, argindex); 
}

#endif
