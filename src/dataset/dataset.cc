// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include <hip/hip_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "dataset/dataset.hpp"

namespace qsr {

Dataset::Dataset(int m, int n) noexcept :
    m(m), n(n) {
    X_d = Array2D<float>(n, m);
    y_d = Array1D<float>(m);
}

Dataset::Dataset(const float *const *X, const float *y, int m, int n) noexcept :
    m(m), n(n) {
    /*
    * The argument X is in array of structures form. 
    * The dimensions are X[m][n]

    * The goal is to convert X to X_d on device, which is in structure of array form.
    * The dimensions should be X_d[n][m]
    */
    X_d = Array2D<float>(n, m);

    // Copy X to X_d on device
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            X_d.ptr[i,j] = X[j][i];
        }
    }

    // Allocate y_d on device
    y_d = Array1D<float>(m);

    // Copy y to y_d on device
    for (int j = 0; j < m; ++j) {
        y_d.ptr[j] = y[j];
    }
}


Dataset::Dataset(pybind11::array_t<float> numpy_X, pybind11::array_t<float> numpy_y) {
    auto numpy_X_buffer_info = numpy_X.request();
    auto numpy_y_buffer_info = numpy_y.request();

    if (numpy_X_buffer_info.ndim != 1 && numpy_X_buffer_info.ndim != 2) {
        throw pybind11::value_error("Argument X (feature matrix) must be an either one- or two-dimensional array.");
    }

    if (numpy_y_buffer_info.ndim != 1) {
        throw pybind11::value_error("Argument y (target vector) must be a one-dimensional array.");
    }

    if (numpy_X_buffer_info.shape[0] != numpy_y_buffer_info.shape[0]) {
        throw pybind11::value_error("First dimensions of X and y must match (the number of data points)");
    }

    // Number of rows / data points
    m = numpy_X_buffer_info.shape[0];

    // Number of columns / features
    n = numpy_X_buffer_info.ndim == 1 ? 1 : numpy_X_buffer_info.shape[1];

    /*
    * The argument X is in array of structures form, implemented as a flat array.
    * The dimensions are X[m * n + n]

    * The goal is to convert X to X_d on device, which is in structure of array form.
    * The dimensions should be X_d[n * m + m]
    */
    X_d = Array2D<float>(n, m);

    // Allocate y_d on device
    y_d = Array1D<float>(m);

    // Allocate and fill X
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            X_d.ptr[i,j] = (static_cast<float*>(numpy_X_buffer_info.ptr))[j * n + i];
        }
    }

    // Allocate and fill y
    for (int j = 0; j < m; ++j) {
        y_d.ptr[j] = (static_cast<float*>(numpy_y_buffer_info.ptr))[j];
    }
}

}