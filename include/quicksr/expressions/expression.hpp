// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef EXPRESSION_H
#define EXPRESSION_H

#include <ostream>
#include <vector>
#include <functional>

#define BINARY_OP_CASE(case_name, operand, operator_token) \
    case case_name: { \
        auto r1 = operand(0); \
        auto r2 = operand(1); \
        return r1 operator_token r2; \
    }

#define UNARY_OP_CASE(case_name, operand, operator_token) \
    case case_name: return operator_token(operand(0))

namespace qsr {

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
    EXPONENTIAL = 9,
    RECTIFIED_LINEAR_UNIT = 10
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
        num_of_nodes(1),
        depth(1) {}

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
        num_of_nodes(1),
        depth(1) {}

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

    /// Depth of the tree representing the expression
    int depth;

    /// Training loss (MSE on dataset)
    float loss;

    /// Fitness (max loss in the population - loss)
    float fitness;

    /// Learned weights
    std::vector<float> weights;
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
  * @brief Tests whether the first expression is less fit than the second
 */
bool operator < (const Expression& left_operand, const Expression& right_operand) noexcept;

/**
  * @brief Tests whether the first expression is more fit than the second
 */
bool operator > (const Expression& left_operand, const Expression& right_operand) noexcept;

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

}

static inline void hash_combine(std::size_t& seed, std::size_t h) noexcept {
    // magic constant from boost::hash_combine
    seed ^= h + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template<>
struct std::hash<qsr::Expression> {
    std::size_t operator()(const qsr::Expression& e) const noexcept {
        std::size_t seed = 0;

        // 1. operation_t – hash its underlying integral value
        hash_combine(seed,
            std::hash<std::underlying_type_t<qsr::operation_t>>{}(
                static_cast<std::underlying_type_t<qsr::operation_t>>(e.operation)));

        // 2. float value – use std::hash<float>
        hash_combine(seed, std::hash<float>{}(e.value));

        // 3. argindex
        hash_combine(seed, std::hash<int>{}(e.argindex));

        // 4. operands (recursive)
        for (const auto& child : e.operands) {
            hash_combine(seed, std::hash<qsr::Expression>{}(child));
        }

        // 5. num_of_nodes
        hash_combine(seed, std::hash<int>{}(e.num_of_nodes));

        // 6. depth
        hash_combine(seed, std::hash<int>{}(e.depth));

        return seed;
    }
};

#endif
