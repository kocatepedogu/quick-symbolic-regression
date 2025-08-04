// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "testdata.hpp"

namespace qsr {

void generate_test_data(float **&X, float *&y, std::function<float(float&)> ground_truth) {
    // Allocate two dimensional feature matrix
    X = new float*[test_data_length];
    for (int i = 0; i < test_data_length; ++i) {
        // Number of input features is one
        X[i] = new float[1];
    }

    // Allocate one dimensional target array
    y = new float[test_data_length];

    // Fill feature matrix with scalar values ranging from 0 to 10
    float x_low = 0;
    float x_high = 10;
    float x_step = (x_high - x_low) / test_data_length;
    for (int i = 0; i < test_data_length; ++i) {
        X[i][0] = x_low + i * x_step;
    }

    // Fill target array with ground truth values
    for (int i = 0; i < test_data_length; ++i) {
        float x = X[i][0];
        y[i] = ground_truth(x);
    }
}

void delete_test_data(float **&X, float *&y) {
    // Deallocate target array
    delete[] y;

    // Deallocate feature matrix
    for (int i = 0; i < test_data_length; ++i) {
        delete[] X[i];
    }
    delete[] X;
}

}