// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "runner.hpp"

namespace qsr::hybrid {
    Runner::Runner(std::shared_ptr<Dataset> dataset, int nweights) :
        dataset(dataset),
        nweights(nweights) {
    }

    void Runner::run(std::vector<Expression>& population, int epochs, float learning_rate) {
        // Implementation for hybrid runner
    }
}