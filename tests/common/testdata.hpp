// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef TESTDATA_HPP
#define TESTDATA_HPP

#include <functional>

namespace qsr {

// Number of data points
constexpr int test_data_length = 128;

void generate_test_data(float **&X, float *&y, std::function<float(float&)> ground_truth);

void delete_test_data(float **&X, float *&y);

}

#endif