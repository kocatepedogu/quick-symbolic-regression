// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef EXPRESSION_GENERATOR_FULL_HPP
#define EXPRESSION_GENERATOR_FULL_HPP

#include "../expressions/expression.hpp"

namespace qsr {

class FullExpressionGenerator {
    public:
        constexpr FullExpressionGenerator(int nvars, int nweights, int depth) : 
            nvars(nvars), nweights(nweights), depth(depth) {
    
            if (depth <= 0) {
                fprintf(stderr, "FullExpressionGenerator: depth must be greater than zero.\n");
                abort();
            }

            if (nvars <= 0) {
                fprintf(stderr, "FullExpressionGenerator: number of variables must be greater than zero.\n");
                abort();
            }
        }

        Expression generate(int remaining_depth) const noexcept;

        Expression generate() const noexcept;

    private:
        const int nvars;
        const int nweights;
        const int depth;

};

}

#endif