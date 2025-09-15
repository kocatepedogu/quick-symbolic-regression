# SPDX-FileCopyrightText: 2025 Doğu Kocatepe
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import matplotlib.pyplot as plt

from quicksr import *

# Define constants
NVARS=2
NPOPULATION=11200
NISLANDS=28
NWEIGHTS=0

# Generate dataset
X = []
for x0 in np.linspace(-5, 5, 64):
    for x1 in np.linspace(-5, 5, 64):
        X.append([x0, x1])

y = [1/(1 + x0**(-4)) + 1/(1 + x1**(-4))
     for x0, x1 in X]

# Create model
model = SymbolicRegressionModel(NVARS, NWEIGHTS, NPOPULATION, NISLANDS, functions=["+", "/", 'cos', 'relu'])

# Fit model
solution, history, time_history = model.fit(X, y, ngenerations=15, nsupergenerations=4, nepochs=0)

# Print best solution
print("Best solution: {}".format(solution))

# Print best loss
print("Best loss: {}".format(solution.loss))

# Plot learning history
plt.figure(figsize=(10, 6))
plt.plot(time_history, history)
plt.xlabel('Time (milliseconds)')
plt.ylabel('Loss (MSE)')
plt.title('Learning History')
plt.grid()
plt.savefig('benchmark_learning_history_pagie_polynomial.png')

# Compute predicted values
y_predicted = model.predict(X)

# Compare target and predicted values
fig = plt.figure(figsize=(15, 15))

ax = fig.add_subplot(111, projection='3d')
ax.scatter(np.array(X)[:, 0], np.array(X)[:, 1], np.array(y), c='b', marker='o', label='Target')
ax.scatter(np.array(X)[:, 0], np.array(X)[:, 1], np.array(y_predicted), c='r', marker='x', label='Prediction')
ax.legend()

plt.title('Target vs Prediction')
plt.savefig('benchmark_target_vs_predicted_pagie_polynomial.png')
plt.show()