# SPDX-FileCopyrightText: 2025 Doğu Kocatepe
# SPDX-License-Identifier: GPL-3.0-or-later

import libquicksr
import numpy as np

from libquicksr import *

class SymbolicRegressionModel:
    def __init__(self, 
        nvars,
        nweights, 
        npopulation,
        nislands,
        elite_rate=0.0,
        migration_rate=0.1,
        max_depth=16,
        initialization=GrowInitialization(), 
        mutation=DistributionMutation([SubtreeMutation(), HoistMutation(), PointMutation()], [0.5, 0.25, 0.25]), 
        recombination=DefaultRecombination(), 
        selection=FitnessProportionalSelection(), 
        runner_generator=HybridRunnerGenerator(),
        functions=['+','-','*', '/', 'sin', 'cos', 'exp', 'relu']):
        """
        Initialize the SymbolicRegressionModel with the given parameters.
        :param nvars: Number of input variables/features.
        :param nweights: Maximum number of trainable weights in candidate expressions.
        :param npopulation: Population size within each island.
        :param nislands: Number of islands in the genetic algorithm.
        :param noffspring: Number of offspring generated per generation.
        :param initialization: Initialization strategy for the genetic algorithm.
        :param mutation: Mutation operator for the genetic algorithm.
        :param recombination: Recombination operator for the genetic algorithm.
        :param selection: Selection strategy for the genetic algorithm.
        :param runner_generator: Runner generator (CPURunnerGenerator, HybridRunnerGenerator, InterIndividualRunnerGenerator, IntraIndividualRunnerGenerator)
        :param functions: List of allowed functions in the candidate expressions.
        """

        # Create model
        self.model = GeneticProgrammingIslands(
            nislands=nislands, 
            config=Config(nvars, nweights, max_depth, npopulation, elite_rate, migration_rate, FunctionSet(functions)),
            toolbox=Toolbox(initialization, mutation, recombination, selection),
            runner_generator=runner_generator
        )

        # Initialize history, solution and compiled solution to None
        self.time_history = None
        self.history = None
        self.solution = None
        self.compiled_solution = None


    def fit(self, X, y, ngenerations, nsupergenerations, nepochs=1, learning_rate=1e-3, verbose=True):
        """
        Fit the model to the given data.
        :param X: Feature matrix (either 1D or 2D array)
        :param y: Target vector
        :param ngenerations: Number of generations to run within each supergeneration.
        :param nsupergenerations: Number of supergenerations to run.
        :param nepochs: Number of epochs in gradient descent.
        :param learning_rate: Learning rate for gradient descent.
        :param verbose: Whether to print progress.
        """
        # Create a dataset that is availabe to both the CPU and the GPU
        dataset = Dataset(X, y)

        # Fit the model
        solution, history, time_history = self.model.fit(dataset, ngenerations, nsupergenerations, nepochs, learning_rate, verbose)

        # Save the solution and history
        self.solution = solution
        self.history = history
        self.time_history = time_history

        # Return the solution and history
        return solution, history, time_history


    def predict(self, X):
        """
        Predict the target values for the given feature matrix.
        :param X: Feature matrix (either 1D or 2D array)
        :return: Predicted target values
        """
        if self.compiled_solution is None:
            self.compiled_solution = self.__lambdify(self.solution)
        return self.compiled_solution(X)


    def __lambdify(self, expr):
        """
        Convert the solution to a lambda function.
        :param expr: Symbolic expression to convert.
        :return: Lambda function representing the expression.
        """

        assert self.solution is not None, "Model has not been fitted yet."

        if len(expr.operands) == 1:
            operand1 = self.__lambdify(expr.operands[0])

        if len(expr.operands) == 2:
            operand1 = self.__lambdify(expr.operands[0])
            operand2 = self.__lambdify(expr.operands[1])

        match expr.operation:
            case Operation.CONSTANT:
                return lambda x: expr.value
            case Operation.PARAMETER:
                return lambda x: self.solution.weights[expr.argindex]
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
                return lambda x: np.sin(operand1(np.array(x)))
            case Operation.COSINE:
                return lambda x: np.cos(operand1(np.array(x)))
            case Operation.EXPONENTIAL:
                return lambda x: np.exp(operand1(np.array(x)))
            case Operation.RECTIFIED_LINEAR_UNIT:
                return lambda x: np.maximum(operand1(np.array(x)), 0)

