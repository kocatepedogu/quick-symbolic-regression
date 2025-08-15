// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "base.hpp"

namespace qsr {
    std::shared_ptr<BaseMutator> BaseMutation::get_mutator(int nvars, int nweights, int max_depth) {
        fprintf(stderr, "Unimplemented base method called.");
        abort();
    }
}