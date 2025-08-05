# SPDX-FileCopyrightText: 2025 Doğu Kocatepe
# SPDX-License-Identifier: GPL-3.0-or-later

import libquicksr
import numpy as np
import matplotlib.pyplot as plt

from libquicksr import *

# Define constants
NPOPULATION=11200
NISLANDS=8
NVARS=1
NWEIGHTS=2

# Generate dataset
X = np.linspace(-5, 5, 25)
y = 3*X + 5*X*X + 7*X*X*X

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
    runner_generator=InterIndividualRunnerGenerator()
)

# Fit model
solution, history = model.fit(ngenerations=15, nsupergenerations=4, nepochs=1, verbose=True)

# Print best solution
print("Best solution: {}".format(solution))

# Print learning history
print("Learning history: {}".format(history))

# Plot learning history
plt.figure(figsize=(10, 6))
plt.plot(history)
plt.xlabel('Generation')
plt.ylabel('Loss (MSE)')
plt.title('Learning History - {}'.format(solution))
plt.xticks(np.arange(0, len(history), 1.0))
plt.grid()
plt.savefig('benchmark_learning_history_1d.png')
