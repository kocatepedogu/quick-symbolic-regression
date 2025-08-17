// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "../genetic/mutation/mutator/subtree_mutator.hpp"
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

    Expression f = w0 + w1*Cos(x) + w2*Sin(y);

    Config config(
        2, 3, 10, 11200, 11200,
        std::make_shared<FunctionSet>(
            std::vector<std::string>{"+", "-", "*", "/", "sin", "cos", "exp", "relu"})
    );

    SubtreeMutator mutation(config, 0.7, 3);

    for (int i = 0; i < 1000; ++i) {
        std::cout << mutation.mutate(f) << std::endl;
    }

    return 0;
}