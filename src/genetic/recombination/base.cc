// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "genetic/recombination/base.hpp"
#include "genetic/recombination/recombiner/base.hpp"
#include <memory>

namespace qsr {
    std::shared_ptr<BaseRecombiner> BaseRecombination::get_recombiner(int max_depth) {
        fprintf(stderr, "Unimplemented base method called.");
        abort();
    }
}