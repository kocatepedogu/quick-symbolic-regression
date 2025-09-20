// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef QUICKSR_INTERINDIVIDUAL_CACHE_H
#define QUICKSR_INTERINDIVIDUAL_CACHE_H

#include "expressions/expression.hpp"

#include <unordered_set>

namespace qsr::inter_individual {

class Cache {
public:
    std::unordered_set<Expression> cache;

    std::vector<Expression> cached_population;
    std::vector<int> cached_indices;

    std::vector<Expression> uncached_population;
    std::vector<int> uncached_indices;

    void read_from_population(const std::vector<Expression>& population);

    void write_to_population(std::vector<Expression>& population) const;

    void save();
};

}

#endif //QUICKSR_INTERINDIVIDUAL_CACHE_H