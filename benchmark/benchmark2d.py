# SPDX-FileCopyrightText: 2025 Doğu Kocatepe
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import matplotlib.pyplot as plt

from quicksr import *

# Define constants
NPOPULATION=11200
NISLANDS=8
NVARS=2
NWEIGHTS=1

# Generate dataset
X = []
for x0 in np.linspace(-5, 5, 64):
    for x1 in np.linspace(-5, 5, 64):
        X.append([x0, x1])

y = [1/(1 + x0**(-4)) + 1/(1 + x1**(-4))
     for x0, x1 in X]

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
solution, history = model.fit(ngenerations=20, nsupergenerations=15, nepochs=1, verbose=True)

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
plt.xticks(np.arange(0, len(history), 15.0))
plt.grid()
plt.savefig('benchmark_learning_history_2d.png')

# Compute predicted values
solution_lambda = expr_to_lambda(solution)
y_predicted = solution_lambda(X)
print("Predicted values: {}".format(y_predicted))

# Compare target and predicted values
fig = plt.figure(figsize=(15, 15))

ax = fig.add_subplot(111, projection='3d')
ax.scatter(np.array(X)[:, 0], np.array(X)[:, 1], np.array(y), c='b', marker='o', label='Target')
ax.scatter(np.array(X)[:, 0], np.array(X)[:, 1], np.array(y_predicted), c='r', marker='x', label='Prediction')
ax.legend()

plt.title('Target vs Prediction')
plt.savefig('benchmark_target_vs_predicted_2d.png')
plt.show()