// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef FULL_INITIALIZER_HPP
#define FULL_INITIALIZER_HPP

#include "base.hpp"
#include "genetic/common/expression_generator.hpp"

namespace qsr {
    class FullInitializer : public BaseInitializer {
    public:
        FullInitializer(const Config &config);

        void initialize(std::vector<Expression>& population) override;

    private:
        const Config config;

        ExpressionGenerator generator;
    };
}

#endif