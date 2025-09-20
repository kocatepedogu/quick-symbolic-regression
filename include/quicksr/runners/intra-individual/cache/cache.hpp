// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef QUICKSR_INTRAINDIVIDUAL_CACHE_H
#define QUICKSR_INTRAINDIVIDUAL_CACHE_H

#include "expressions/expression.hpp"

#include <unordered_set>

namespace qsr::intra_individual {

class Cache {
public:
    std::unordered_set<Expression> cache;

    bool load(Expression& expression) const;

    void save(const Expression& expression);
};

}

#endif //QUICKSR_INTRAINDIVIDUAL_CACHE_H