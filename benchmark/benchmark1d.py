# SPDX-FileCopyrightText: 2025 Doğu Kocatepe
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import matplotlib.pyplot as plt

from quicksr import *

# Define constants
NPOPULATION=11200
NISLANDS=8
NVARS=1
NWEIGHTS=2

# Generate dataset
X = np.linspace(-5, 5, 25)
y = 1.0 + 3.0*X + 5.0*X*X + 7.0*X*X*X

# Create model
model = GeneticProgrammingIslands(
    Dataset(X, y), 
    nislands=NISLANDS, 
    nweights=NWEIGHTS, 
    npopulation=NPOPULATION, 
    initialization=DefaultInitialization(),
    mutation=DefaultMutation(),
    recombination=DefaultRecombination(),
    selection=FitnessProportionalSelection(),
    runner_generator=HybridRunnerGenerator()
)

# Fit model
solution, history = model.fit(ngenerations=15, nsupergenerations=4, nepochs=500, verbose=True)

# Print best solution
print("Best solution: {}".format(solution))

# Print learning history
print("Learning history: {}".format(history))

# Plot learning history
plt.figure(figsize=(10, 6))
plt.plot(history)
plt.xlabel('Generation')
plt.ylabel('Loss (MSE)')
plt.title('Learning History')
plt.xticks(np.arange(0, len(history), 5.0))
plt.grid()
plt.savefig('benchmark_learning_history_1d.png')

# Compute predicted values
solution_lambda = expr_to_lambda(solution)
y_predicted = solution_lambda(X)
print("Predicted values: {}".format(y_predicted))

# Compare target and predicted values
plt.figure(figsize=(10, 6))
plt.plot(X, y, label='Target')
plt.plot(X, y_predicted, label='Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Target vs Predicted')
plt.legend()
plt.grid()
plt.savefig('benchmark_target_vs_predicted_1d.png')