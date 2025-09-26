import time
import torch
import numpy as np

from evogp.tree import Forest, GenerateDescriptor
from evogp.algorithm import (
    GeneticProgramming,
    DefaultSelection,
    DefaultMutation,
    DefaultCrossover,
)
from evogp.problem import SymbolicRegression, BaseProblem
from evogp.pipeline import StandardPipeline

from benchmark_utils import run_benchmark


class RecordedPipeline(StandardPipeline):
    """
    Overrides run method to record loss history with respect to time in evoGP model training.
    """
    def __init__(
            self,
            algorithm: GeneticProgramming,
            problem: BaseProblem,
            fitness_target: float = None,
            generation_limit: int = 100,
            time_limit: int = None,
            is_show_details: bool = True,
            valid_fitness_boundry: float = 1e8,
    ):
        super().__init__(
            algorithm,
            problem,
            fitness_target,
            generation_limit,
            time_limit,
            is_show_details,
            valid_fitness_boundry)

        self.fitness_history = None
        self.time_history = None

    def run(self):
        tic = time.time()

        self.time_history = []
        self.fitness_history = []

        generation_cnt = 0
        while True:

            if self.is_show_details:
                start_time = time.time()

            cpu_fitness = self.step()

            self.time_history.append(time.time() - tic)
            self.fitness_history.append(self.best_fitness)

            if self.is_show_details:
                self.show_details(start_time, generation_cnt, cpu_fitness)
                print(self.best_tree)

            if (
                    self.fitness_target is not None
                    and self.best_fitness >= self.fitness_target
            ):
                print("Fitness target reached!")
                break

            if self.time_limit is not None and time.time() - tic > self.time_limit:
                print("Time limit reached!")
                break

            generation_cnt += 1
            if generation_cnt >= self.generation_limit:
                print("Generation limit reached!")
                break

        return self.best_tree


def solve_with_evogp(config, X, y):
    """
    Runs symbolic regression on the dataset using evoGP
    :param config: Algorithm configuration
    :param X: features
    :param y: target
    """

    # Move dataset to GPU
    train_X = torch.tensor(
        X,
        dtype=torch.float,
        device='cuda'
    )
    train_y = torch.tensor(
        y,
        dtype=torch.float,
        device='cuda'
    )

    problem = SymbolicRegression(datapoints=train_X, labels=train_y)

    # create decriptor for generating new trees
    descriptor = GenerateDescriptor(
        max_tree_len=2**config['MAX_DEPTH'],
        input_len=problem.problem_dim,
        output_len=problem.solution_dim,
        using_funcs=["+", "*", "cos"],
        max_layer_cnt=config['MAX_DEPTH'],
        const_samples=[-1, 0, 1],
    )

    # create the algorithm
    algorithm = GeneticProgramming(
        initial_forest=Forest.random_generate(pop_size=config['POPULATION_SIZE'], descriptor=descriptor),
        crossover=DefaultCrossover(),
        mutation=DefaultMutation(
            mutation_rate=config['MUTATION_RATE'], descriptor=descriptor.update(max_layer_cnt=config['MUTATION_DEPTH_INCREMENT'])
        ),
        selection=DefaultSelection(survival_rate=config['SURVIVAL_RATE'], elite_rate=config['ELITE_RATE']),
        enable_pareto_front=False,
    )

    pipeline = RecordedPipeline(
        algorithm,
        problem,
        generation_limit=config['GENERATION_LIMIT']+1,
    )

    best = pipeline.run()

    pred_res = best.forward(train_X)
    print(torch.sum((pred_res - train_y)**2)/len(X))

    sympy_expression = best.to_sympy_expr()
    print("Result:", sympy_expression)

    loss_wrt_gen = np.abs(pipeline.fitness_history)
    loss_wrt_time = np.abs(pipeline.fitness_history)
    time_hist = np.array(pipeline.time_history)

    return loss_wrt_gen, loss_wrt_time, time_hist - np.min(time_hist)


if __name__ == '__main__':
    #torch.manual_seed(42)
    #np.random.seed(42)
    #random.seed(42)
    run_benchmark(solve_with_evogp)
