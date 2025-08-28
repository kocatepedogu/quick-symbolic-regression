// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "runners/runner_generator_base.hpp"

#include <cstdio>

namespace qsr {

std::shared_ptr<BaseRunner> BaseRunnerGenerator::generate(int nweights) {
    fprintf(stderr, "Unimplemented base method called.");
    abort();
}

}