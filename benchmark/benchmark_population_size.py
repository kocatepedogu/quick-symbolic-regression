# SPDX-FileCopyrightText: 2025 Doğu Kocatepe
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import matplotlib.pyplot as plt

from quicksr import *

# Define constants
NISLANDS=28
NWEIGHTS=2

# Generate dataset
X = np.linspace(-5, 5, 25)
y = 3*X + 5*X*X + 7*X*X*X

POPULATIONS = [
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
]

def fit_model(npopulation, runner_generator):
    import time
    start_time = time.time()

    # Create model
    model = SymbolicRegressionModel(
        nvars=1,
        nweights=NWEIGHTS, 
        npopulation=npopulation, 
        nislands=NISLANDS,
        runner_generator=runner_generator
    )

    # Fit model
    solution, loss_wrt_gen, loss_wrt_time, time_hist = model.fit(X, y, ngenerations=2, nsupergenerations=2, nepochs=250)

    # Find elapsed time
    elapsed_time = time.time() - start_time

    # Return MSE and elapsed time
    return loss_wrt_time[-1], elapsed_time


mse_values_cpu = []
elapsed_times_cpu = []

mse_values_hybrid = []
elapsed_times_hybrid = []


for pop_per_island in POPULATIONS:
    npopulation = pop_per_island * NISLANDS

    mse, elapsed_time = fit_model(npopulation, CPURunnerGenerator())
    print(f'CPU - Population: {npopulation}, MSE: {mse}, Elapsed Time: {elapsed_time} seconds')
    mse_values_cpu.append(mse)
    elapsed_times_cpu.append(elapsed_time)

    mse, elapsed_time = fit_model(npopulation, HybridRunnerGenerator())
    print(f'Hybrid - Population: {npopulation}, MSE: {mse}, Elapsed Time: {elapsed_time} seconds')
    mse_values_hybrid.append(mse)
    elapsed_times_hybrid.append(elapsed_time)

# Plot MSE vs Population Size
plt.figure(figsize=(10, 6))
plt.plot(POPULATIONS, mse_values_cpu, label='CPU')
plt.plot(POPULATIONS, mse_values_hybrid, label='Hybrid')
plt.xlabel('Population Size')
plt.ylabel('Loss (MSE)')
plt.title('MSE vs Population Size')
plt.legend()
plt.grid()
plt.savefig('benchmark_mse_vs_population_size.png')

# Plot Elapsed Time vs Population Size
plt.figure(figsize=(10, 6))
plt.plot(POPULATIONS, elapsed_times_cpu, label='CPU')
plt.plot(POPULATIONS, elapsed_times_hybrid, label='Hybrid')
plt.xlabel('Population Size')
plt.ylabel('Elapsed Time (seconds)')
plt.title('Elapsed Time vs Population Size')
plt.legend()
plt.grid()
plt.savefig('benchmark_elapsed_time_vs_population_size.png')
