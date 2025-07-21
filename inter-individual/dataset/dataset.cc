// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "dataset.hpp"

#include "../../util.hpp"

namespace inter_individual {
    Dataset::Dataset(const float *const *X, const float *y, int m, int n) noexcept : m(m), n(n) {
        /*
        * The argument X is in array of structures form. 
        * The dimensions are X[m][n]

        * The device array X_d is also going to be in array of structures form.
        * The dimensions should be X_d[m][n]
        */
        HIP_CALL(hipMallocManaged(&X_d, sizeof *X_d * m));
        for (int i = 0; i < m; ++i) {
            HIP_CALL(hipMallocManaged(&X_d[i], sizeof **X_d * n));
        }

        // Copy X to X_d on device
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                X_d[i][j] = X[i][j];
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
        for (int i = 0; i < m; ++i) {
            HIP_CALL(hipFree(X_d[i]));
        }
        HIP_CALL(hipFree(X_d));
    }
};