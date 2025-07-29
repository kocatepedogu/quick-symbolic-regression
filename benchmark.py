# SPDX-FileCopyrightText: 2025 Doğu Kocatepe
# SPDX-License-Identifier: GPL-3.0-or-later

import libquicksr
import numpy as np
import matplotlib.pyplot as plt

from libquicksr import *

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

y = [x0*np.exp(x0 + x1) + np.sin(x1) 
     for x0, x1 in X]

# Create model
model = GeneticProgrammingIslands(
    Dataset(X, y), 
    nislands=NISLANDS, 
    nweights=NWEIGHTS, 
    npopulation=NPOPULATION, 
    initializer=DefaultInitializer(nvars=NVARS, nweights=NWEIGHTS, npopulation=NPOPULATION//NISLANDS),
    mutation=DefaultMutation(nvars=NVARS, nweights=NWEIGHTS),
    crossover=DefaultCrossover(),
    selection=FitnessProportionalSelection(),
    runner_generator=IntraIndividualRunnerGenerator()
)

# Fit model
solution, history = model.fit(ngenerations=1, nsupergenerations=5, verbose=True)

# Print best solution
print("Best solution: {}".format(solution))

# Plot learning history
plt.figure(figsize=(10, 6))
plt.plot(history)
plt.xlabel('Generation')
plt.ylabel('Loss (MSE)')
plt.title('Learning History - {}'.format(solution))
plt.grid()
plt.savefig('benchmark_learning_history.png')
