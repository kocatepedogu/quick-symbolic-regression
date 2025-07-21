// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "./tests/testdata.hpp"

#include "./expressions/expression.hpp"
#include "./expressions/binary.hpp"
#include "./expressions/unary.hpp"
#include "inter-individual/dataset/dataset.hpp"

#include <cmath>
#include <iostream>

int main(void) {
    float **X, *y;

    // Generate ground truth data
    generate_test_data(X, y, [](float x) { 
        return 2.5382 * cos(x)*x + x*x - 0.5; 
    });

    // Create dataset
    inter_individual::Dataset dataset(X, y, test_data_length, 1);

    // Free data
    delete_test_data(X, y);

    return 0;
}