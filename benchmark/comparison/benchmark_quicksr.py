# SPDX-FileCopyrightText: 2025 Doğu Kocatepe
# SPDX-License-Identifier: GPL-3.0-or-later
import time

import matplotlib.pyplot as plt
import numpy as np
import quicksr as qsr
from libquicksr import *

from benchmark_utils import *

def solve_with_quicksr(config, X, y):
    """
    Runs symbolic regression on the dataset using QuickSR
    :param config: Algorithm configuration
    :param X: features
    :param y: target
    """

    start_time = time.time()

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y).reshape(-1)

    # Create model
    model = qsr.SymbolicRegressionModel(
        nvars=X.shape[1],
        nweights=0,
        npopulation=config['POPULATION_SIZE'],
        nislands=config['NUMBER_OF_ISLANDS'],
        survival_rate=config['SURVIVAL_RATE'],
        elite_rate=config['ELITE_RATE'],
        migration_rate=config['MIGRATION_RATE'],
        functions=["+", "-", "*", "cos"],
        mutation=qsr.SubtreeMutation(mutation_probability=config['MUTATION_RATE'], max_depth_increment=config['MUTATION_DEPTH_INCREMENT']),
        recombination=qsr.DefaultRecombination(crossover_probability=1.0),
        initialization=qsr.RampedHalfAndHalfInitialization(init_depth=config['MAX_DEPTH']),
        selection=qsr.RankSelection(sp=0.0),
        max_depth=config['MAX_DEPTH'],
        runner_generator=HybridRunnerGenerator(use_cache=config['USE_CACHE']),
        enable_parsimony_pressure=False)

    # Fit model
    super_generations = config['GENERATION_LIMIT']
    solution, loss_wrt_gen, loss_wrt_time, time_hist = model.fit(
        X, y, ngenerations=config['GENERATION_LIMIT'] // super_generations, nsupergenerations=super_generations, nepochs=0)

    # Print best solution
    print("Best solution: {}".format(solution))

    # Print best loss
    print("Best loss: {}".format(solution.loss))

    # Visualize result
    y_predicted = model.predict(X)
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(np.array(X)[:, 0], np.array(X)[:, 1], np.array(y), c='b', marker='o', label='Target')
    ax.scatter(np.array(X)[:, 0], np.array(X)[:, 1], np.array(y_predicted), c='r', marker='x', label='Prediction')
    ax.legend()
    plt.title('Target vs Prediction')
    plt.savefig('benchmark_target_vs_predicted_pagie_polynomial.png')

    loss_wrt_gen = np.array([loss_wrt_gen[0]] + loss_wrt_gen)
    loss_wrt_time = np.array([loss_wrt_time[0]] + loss_wrt_time)

    time_hist = np.array([start_time * 1000.0] + time_hist) / 1000.0
    time_hist -= np.min(time_hist)

    return loss_wrt_gen, loss_wrt_time, time_hist

if __name__ == '__main__':
    run_benchmark(solve_with_quicksr)
