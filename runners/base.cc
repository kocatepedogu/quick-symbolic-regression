// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "base.hpp"

#include <cstdio>

namespace qsr {

void BaseRunner::run(std::vector<Expression>& population, int epochs, float learning_rate) {
    fprintf(stderr, "Unimplemented base method called.");
    abort();
}

}