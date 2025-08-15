// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "base.hpp"
#include "recombiner/base.hpp"
#include <memory>

namespace qsr {
    std::shared_ptr<BaseRecombiner> BaseRecombination::get_recombiner(int max_depth) {
        fprintf(stderr, "Unimplemented base method called.");
        abort();
    }
}