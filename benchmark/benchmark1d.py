# SPDX-FileCopyrightText: 2025 Doğu Kocatepe
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import matplotlib.pyplot as plt

from quicksr import *

# Define constants
NVARS=1
NPOPULATION=11200
NWEIGHTS=2
NISLANDS=28

# Generate dataset
X = np.linspace(-5, 5, 25)
y = 1.0 + 3.0*X + 5.0*X*X + 7.0*X*X*X

# Create model
model = SymbolicRegressionModel(NVARS, NWEIGHTS, NPOPULATION, NISLANDS, initialization=GrowInitialization(init_depth=1))

# Fit model
solution, history, time_history = model.fit(X, y, ngenerations=15, nsupergenerations=4, nepochs=500)

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
plt.savefig('benchmark_learning_history_1d.png')

# Compute predicted values
y_predicted = model.predict(X)

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
