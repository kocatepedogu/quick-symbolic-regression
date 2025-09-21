// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef INTRA_DATASET_HPP
#define INTRA_DATASET_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "util/arrays/array1d.hpp"
#include "util/arrays/array2d.hpp"

namespace qsr {

struct Dataset {
    /**
     * @brief Constructs an empty dataset on GPU
     **/
    Dataset(int m, int n) noexcept;

    /**
      * @brief Constructs a dataset on GPU from given X matrix and y vector
      */
    Dataset(const float *const *X, const float *y, int m, int n) noexcept;

    /**
      * @brief Constructs a dataset on GPU from given X numpy matrix and y numpy vector
     */
    Dataset(pybind11::array_t<float> numpy_X, pybind11::array_t<float> numpy_y);

    /// Number of data points
    int m;

    /// Number of input features (dimensions)
    int n;

    /// Input feature matrix on device memory
    Array2D<float> X_d;

    /// Target variable on device memory
    Array1D<float> y_d;
};

}

#endif