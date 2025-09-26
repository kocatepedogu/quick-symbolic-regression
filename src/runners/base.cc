// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "runners/base.hpp"

#include "util/precision.hpp"

#include <cstdio>
#include <memory>

using namespace qsr;

BaseRunner::BaseRunner(int nweights) : 
    nweights(nweights) {}

void BaseRunner::resize_arrays(int stack_length, int intermediate_length, int nproblem, int nthreads) {
    loss_d.resize(nproblem);
    weights_grad_d.resize(nweights, nproblem);
    stack_d.resize(stack_length, nthreads);
    intermediate_d.resize(intermediate_length, nthreads);
}

void BaseRunner::run(std::vector<Expression>& population, std::shared_ptr<const Dataset> dataset, int epochs, double learning_rate) {
    fprintf(stderr, "Unimplemented base method called.");
    abort();
}
