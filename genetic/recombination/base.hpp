// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef RECOMBINATION_BASE_HPP
#define RECOMBINATION_BASE_HPP

#include "recombiner/base.hpp"
#include <memory>

namespace qsr {
    class BaseRecombination {
        public:
            virtual std::shared_ptr<BaseRecombiner> get_recombiner(int max_depth);

            virtual ~BaseRecombination() = default;
    };
}

#endif