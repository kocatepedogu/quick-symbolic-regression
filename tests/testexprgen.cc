// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "../genetic/expression_generator.hpp"

#include <iostream>

int main(void) {
    ExpressionGenerator expression_generator(2, 3, 4);

    for (int i = 0; i < 1000; i++) {
        std::cout << expression_generator.generate() << std::endl;
    }

    return 0;
}