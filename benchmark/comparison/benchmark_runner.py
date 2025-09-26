# SPDX-FileCopyrightText: 2025 Doğu Kocatepe
# SPDX-License-Identifier: GPL-3.0-or-later

import matplotlib.pyplot as plt
from benchmark_utils import *

if __name__ == '__main__':
    config = {
        # Main Parameters
        'POPULATION_SIZE': 11200,
        'MAX_DEPTH': 8,
        'GENERATION_LIMIT': 60,

        # Common Genetic Programming Parameters
        'MUTATION_RATE': 0.2,
        'MUTATION_DEPTH_INCREMENT': 3,
        'SURVIVAL_RATE': 0.3,
        'ELITE_RATE': 0.01,

        # QuickSR-specific Parameters
        'NUMBER_OF_ISLANDS': 10,
        'MIGRATION_RATE': 0.1,
        'USE_CACHE': True
    }

    # Generate Pagie Polynomial dataset

    X = []
    for x0 in np.linspace(-5, 5, 64):
        for x1 in np.linspace(-5, 5, 64):
            X.append([x0, x1])

    y = [[1 / (1 + x0 ** -4) + 1 / (1 + x1 ** -4)] for x0, x1 in X]

    # Run EvoGP and QuickSR

    run_queue = []
    for i in range(10):
        run_queue.append(('evoGP', 'blue', 'benchmark_evogp.py', config))
    for i in range(10):
        run_queue.append(('QuickSR', 'red', 'benchmark_quicksr.py', config))

    run_results = [run_benchmark_script(script_name, config, X, y)
                   for label, color, script_name, config in run_queue]

    # Plot evolution with respect to time

    plt.figure(figsize=(15, 15))
    plt.ylim(0, 0.2)
    plt.xlim(0, 7)
    for queue_element, result in zip(run_queue, run_results):
        label, color, script_name, config = queue_element
        loss_wrt_gen, loss_wrt_time, time_hist = result
        plt.plot(time_hist, loss_wrt_time, label=label, c=color)
    plt.legend()
    plt.title('Evolution w.r.t Time')
    plt.savefig('evolution_wrt_time.png')

    # Plot evolution with respect to generation

    plt.figure(figsize=(15, 15))
    plt.xlim(0, 7)
    plt.ylim(0, 0.2)
    for queue_element, result in zip(run_queue, run_results):
        label, color, script_name, config = queue_element
        loss_wrt_gen, loss_wrt_time, time_hist = result
        plt.plot(loss_wrt_gen, label=label, c=color)
    plt.legend()
    plt.title('Evolution w.r.t Generation')
    plt.savefig('evolution_wrt_gen.png')
