#ifndef MACRO_HPP
#define MACRO_HPP

#define BINARY_OP_CASE(case_name, operand, operator_token) \
    case case_name: { \
        auto r1 = operand(0); \
        auto r2 = operand(1); \
        return r1 operator_token r2; \
    }

#define UNARY_OP_CASE(case_name, operand, operator_token) \
    case case_name: return operator_token(operand(0))

#endif