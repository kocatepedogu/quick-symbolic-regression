// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "base.hpp"

#include <stdio.h>
#include <stdlib.h>

namespace qsr {

void BaseInitializer::initialize(std::vector<Expression>& population) {
    fprintf(stderr, "Unimplemented base method called.");
    abort();
}

}