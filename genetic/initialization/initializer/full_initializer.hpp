// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef FULL_INITIALIZER_HPP
#define FULL_INITIALIZER_HPP

#include "base.hpp"

#include "../../common/expression_generator_full.hpp"

namespace qsr {
    class FullInitializer : public BaseInitializer {
    public:
        FullInitializer(int nvars, int nweights, int depth, int npopulation, std::shared_ptr<FunctionSet> function_set);

        void initialize(std::vector<Expression>& population) override;

    private:
        const int npopulation;

        FullExpressionGenerator generator;
    };
}

#endif