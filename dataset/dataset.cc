// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include <hip/hip_runtime.h>

#include "dataset.hpp"
#include "../util.hpp"


Dataset::Dataset(const float *const *X, const float *y, int m, int n) noexcept : 
    m(m), n(n) {
    /*
     * The argument X is in array of structures form. 
     * The dimensions are X[m][n]

     * The goal is to convert X to X_d on device, which is in structure of array form.
     * The dimensions should be X_d[n][m]
     */
    HIP_CALL(hipMallocManaged(&X_d, sizeof *X_d * n));
    for (int i = 0; i < n; ++i) {
        HIP_CALL(hipMallocManaged(&X_d[i], sizeof **X_d * m));
    }

    // Copy X to X_d on device
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            X_d[i][j] = X[j][i];
        }
    }

    // Allocate y_d on device
    HIP_CALL(hipMallocManaged(&y_d, sizeof *y_d * m));

    // Copy y to y_d on device
    for (int j = 0; j < m; ++j) {
        y_d[j] = y[j];
    }
}


Dataset::~Dataset() noexcept {
    // Delete y_d from device
    HIP_CALL(hipFree(y_d));

    // Delete X_d from device
    for (int i = 0; i < n; ++i) {
        HIP_CALL(hipFree(X_d[i]));
    }
    HIP_CALL(hipFree(X_d));
}