
import libquicksr
import numpy as np

from libquicksr import *

def expr_to_lambda_recurse(expr, root):
    if len(expr.operands) == 1:
        operand = expr_to_lambda_recurse(expr.operands[0], root)

    if len(expr.operands) == 2:
        operand1 = expr_to_lambda_recurse(expr.operands[0], root)
        operand2 = expr_to_lambda_recurse(expr.operands[1], root)

    match expr.operation:
        case Operation.CONSTANT:
            return lambda x: expr.value
        case Operation.PARAMETER:
            return lambda x: root.weights[expr.argindex]
        case Operation.IDENTITY:
            return lambda x: np.array(x) if len(np.array(x).shape) == 1 else np.array(x)[:, expr.argindex]
        case Operation.ADDITION:
            return lambda x: operand1(np.array(x)) + operand2(np.array(x))
        case Operation.SUBTRACTION:
            return lambda x: operand1(np.array(x)) - operand2(np.array(x))
        case Operation.MULTIPLICATION:
            return lambda x: operand1(np.array(x)) * operand2(np.array(x))
        case Operation.DIVISION:
            return lambda x: operand1(np.array(x)) / operand2(np.array(x))
        case Operation.SINE:
            return lambda x: np.sin(operand(np.array(x)))
        case Operation.COSINE:
            return lambda x: np.cos(operand(np.array(x)))
        case Operation.EXPONENTIAL:
            return lambda x: np.exp(operand(np.array(x)))

def expr_to_lambda(expr):
    return expr_to_lambda_recurse(expr, expr)