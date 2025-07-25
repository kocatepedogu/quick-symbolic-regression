# SPDX-FileCopyrightText: 2025 Doğu Kocatepe
# SPDX-License-Identifier: GPL-3.0-or-later

import libquicksr
import numpy as np

X = []
for x0 in np.linspace(-5, 5, 64):
    for x1 in np.linspace(-5, 5, 64):
        X.append([x0, x1])

y = [x0*np.exp(x0 + x1) + np.sin(x1) 
     for x0, x1 in X]

libquicksr.fit(
    X, y, 
    nthreads=20, 
    nweights=1, 
    npopulation=11200, 
    ngenerations=1,
    nsupergenerations=5)
