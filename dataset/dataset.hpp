// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef INTRA_DATASET_HPP
#define INTRA_DATASET_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

struct Dataset {
    /**
      * @brief Constructs a dataset on GPU from given X matrix and y vector
      */
    Dataset(const float *const *X, const float *y, int m, int n) noexcept;

    /**
      * @brief Constructs a dataset on GPU from given X numpy matrix and y numpy vector
     */
    Dataset(pybind11::array_t<float> numpy_X, pybind11::array_t<float> numpy_y);

    /**
      * @brief Deletes dataset from GPU memory
      */
    ~Dataset() noexcept;

    /// Number of data points
    int m;

    /// Number of input features (dimensions)
    int n;

    /// Input feature matrix on device memory
    float **X_d;

    /// Target variable on device memory
    float *y_d;
};

#endif