// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef DEFAULT_INITIALIZER_HPP
#define DEFAULT_INITIALIZER_HPP

#include "base.hpp"

#include "../expression_generator.hpp"

class DefaultInitializer : public BaseInitializer {
public:
    constexpr DefaultInitializer(int nvars, int nweights, int max_depth, int npopulation) :
        generator(nvars, nweights, max_depth), npopulation(npopulation) {}

    void initialize(std::vector<Expression>& population) override;

private:
    const int npopulation;

    const ExpressionGenerator generator;
};

#endif