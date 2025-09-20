// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "runners/intra-individual/cache/cache.hpp"

namespace qsr::intra_individual {

bool Cache::load(Expression& expression) const {
    if (cache.contains(expression) && *cache.find(expression) == expression) {
        expression = *cache.find(expression);
        return true;
    }
    return false;
}

void Cache::save(const Expression& expression) {
    cache.insert(expression);
}

}