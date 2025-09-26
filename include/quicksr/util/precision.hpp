// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef QUICKSR_PRECISION_H
#define QUICKSR_PRECISION_H

#include <cmath>
#include "half.hpp"

/* Concept definitions for real numbers and integers */

template <typename  T>
concept Real =
    std::is_same_v<T, __half> ||
    std::is_same_v<T, float> ||
    std::is_same_v<T, double>;

template <typename  T>
concept Integral =
    std::is_same_v<T, int> ||
    std::is_same_v<T, long> ||
    std::is_same_v<T, unsigned int> ||
    std::is_same_v<T, unsigned long> ||
    std::is_same_v<T, size_t>;

/* Definition of real number used in runners */

typedef float real;
static_assert(Real<real>);

/* Literal operators to define constants of type "real" */

constexpr __device__ __host__ real operator""_r(const long double arg) {
    return arg;
}

constexpr __device__ __host__ real operator""_r(const unsigned long long arg) {
    return arg;
}

#endif //QUICKSR_PRECISION_H