// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef DATASET_HPP
#define DATASET_HPP

struct Dataset {
    /**
      * @brief Constructs a dataset on GPU from given X matrix and y vector
      */
    Dataset(const float *const *X, const float *y, int m, int n) noexcept;

    /**
      * @brief Deletes dataset from GPU memory
      */
    ~Dataset() noexcept;

    /// Number of data points
    const int m;

    /// Number of input features (dimensions)
    const int n;

    /// Input feature matrix on device memory
    float **X_d;

    /// Target variable on device memory
    float *y_d;
};

#endif