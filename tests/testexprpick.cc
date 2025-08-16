// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "../genetic/common/expression_picker.hpp"

#include "../expressions/expression.hpp"
#include "../expressions/binary.hpp"
#include "../expressions/unary.hpp"

#include <iostream>

using namespace qsr;

int main(void) {
    ExpressionPicker expression_picker;

    // Input feature
    Expression x = Var(0);

    // Trainable parameters
    Expression w0 = Parameter(0);
    Expression w1 = Parameter(1);
    Expression w2 = Parameter(2);

    // Symbolic expression
    Expression f = (((w0 * Cos(x))*x) + x*x) - w1;

    for (int i = 0; i < 100; i++) {
        std::cout << expression_picker.pick(f) << std::endl;
    }

    return 0;
}