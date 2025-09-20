# SPDX-FileCopyrightText: 2025 Doğu Kocatepe
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import matplotlib.pyplot as plt

from quicksr import *

# Define constants
NPOPULATION=1000
NISLANDS=28
NWEIGHTS=2

DATASET_SIZES = [
    10,
    20,
    50,
    100,
    200,
    500,
    1000,
    2000,
    5000,
    10000,
    20000,
    50000,
    100000,
]

def fit_model(dataset_size, runner_generator):
    import time
    start_time = time.time()

    # Generate dataset
    X = np.linspace(-5, 5, dataset_size)
    y = 3*X + 5*X*X + 7*X*X*X

    # Create model
    model = SymbolicRegressionModel(
        nvars=1, 
        nweights=NWEIGHTS, 
        npopulation=NPOPULATION, 
        nislands=NISLANDS, 
        runner_generator=runner_generator
    )

    # Fit model
    solution, loss_wrt_gen, loss_wrt_time, time_hist = model.fit(X, y, ngenerations=2, nsupergenerations=2, nepochs=100, verbose=True)

    # Find elapsed time
    elapsed_time = time.time() - start_time

    # Return MSE and elapsed time
    return loss_wrt_time[-1], elapsed_time


mse_values_cpu = []
elapsed_times_cpu = []

mse_values_intraindividual = []
elapsed_times_intraindividual = []


for dataset_size in DATASET_SIZES:
    mse, elapsed_time = fit_model(dataset_size, CPURunnerGenerator())
    print(f'CPU - Dataset Size: {dataset_size}, MSE: {mse}, Elapsed Time: {elapsed_time} seconds')
    mse_values_cpu.append(mse)
    elapsed_times_cpu.append(elapsed_time)

    mse, elapsed_time = fit_model(dataset_size, IntraIndividualRunnerGenerator())
    print(f'Intra Individual - Dataset Size: {dataset_size}, MSE: {mse}, Elapsed Time: {elapsed_time} seconds')
    mse_values_intraindividual.append(mse)
    elapsed_times_intraindividual.append(elapsed_time)

# Plot MSE vs Dataset Size
plt.figure(figsize=(10, 6))
plt.plot(DATASET_SIZES, mse_values_cpu, label='CPU')
plt.plot(DATASET_SIZES, mse_values_intraindividual, label='Intra Individual')
plt.xlabel('Dataset Size')
plt.ylabel('Loss (MSE)')
plt.title('MSE vs Dataset Size')
plt.legend()
plt.grid()
plt.savefig('benchmark_mse_vs_dataset_size.png')

# Plot Elapsed Time vs Dataset Size
plt.figure(figsize=(10, 6))
plt.plot(DATASET_SIZES, elapsed_times_cpu, label='CPU')
plt.plot(DATASET_SIZES, elapsed_times_intraindividual, label='Intra Individual')
plt.xlabel('Dataset Size')
plt.ylabel('Elapsed Time (seconds)')
plt.title('Elapsed Time vs Dataset Size')
plt.legend()
plt.grid()
plt.savefig('benchmark_elapsed_time_vs_dataset_size.png')