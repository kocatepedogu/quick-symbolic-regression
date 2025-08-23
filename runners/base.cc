// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "base.hpp"

#include <cstdio>
#include <memory>

using namespace qsr;

BaseRunner::BaseRunner(int nweights) : 
    nweights(nweights) {}

void BaseRunner::run(std::vector<Expression>& population, std::shared_ptr<const Dataset> dataset, int epochs, float learning_rate) {
    fprintf(stderr, "Unimplemented base method called.");
    abort();
}
