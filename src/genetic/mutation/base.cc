// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "genetic/mutation/base.hpp"

namespace qsr {
    std::shared_ptr<BaseMutator> BaseMutation::get_mutator(const Config &config) {
        fprintf(stderr, "Unimplemented base method called.");
        abort();
    }
}