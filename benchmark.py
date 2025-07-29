# SPDX-FileCopyrightText: 2025 Doğu Kocatepe
# SPDX-License-Identifier: GPL-3.0-or-later

import libquicksr
import numpy as np

from libquicksr import *

NPOPULATION=11200
NVARS=2
NWEIGHTS=1

X = []
for x0 in np.linspace(-5, 5, 64):
    for x1 in np.linspace(-5, 5, 64):
        X.append([x0, x1])

y = [x0*np.exp(x0 + x1) + np.sin(x1) 
     for x0, x1 in X]

model = GeneticProgrammingIslands(
    Dataset(X, y), 
    nislands=8, 
    nweights=NWEIGHTS, 
    npopulation=NPOPULATION, 
    ngenerations=1,
    nsupergenerations=5,
    initializer=DefaultInitializer(nvars=NVARS, nweights=NWEIGHTS, npopulation=NPOPULATION//8),
    mutation=DefaultMutation(nvars=NVARS, nweights=NWEIGHTS),
    crossover=DefaultCrossover(),
    selection=FitnessProportionalSelection(),
    runner_generator=IntraIndividualRunnerGenerator()
)

solution = model.fit()
print("Best solution: {}".format(solution))
