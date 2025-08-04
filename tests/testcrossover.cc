// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "../genetic/crossover/default.hpp"
#include "../expressions/unary.hpp"
#include "../expressions/binary.hpp"

#include <iostream>

using namespace qsr;

int main(void) {
    Expression x = Var(0);
    Expression y = Var(1);

    Expression w0 = Parameter(0);
    Expression w1 = Parameter(1);
    Expression w2 = Parameter(2);

    Expression f1 = x * x + w0;
    Expression f2 = Sin(w1 * x);

    DefaultCrossover crossover(1.0);

    for (int i = 0; i < 1000; ++i) {
        auto offspring = crossover.crossover(f1, f2);
        std::cout << std::get<0>(offspring) << " " << std::get<1>(offspring) << std::endl;
    }

    return 0;
}