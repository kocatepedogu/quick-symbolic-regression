// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef RAMPED_HALF_AND_HALF_INITIALIZER_HPP
#define RAMPED_HALF_AND_HALF_INITIALIZER_HPP

#include "base.hpp"
#include "genetic/common/expression_generator.hpp"

namespace qsr {

class RampedHalfAndHalfInitializer : public BaseInitializer {
public:
    RampedHalfAndHalfInitializer(const Config &config);
    void initialize(std::vector<Expression>& population) override;

private:
    const Config config;
    ExpressionGenerator generator;
};

}

#endif
