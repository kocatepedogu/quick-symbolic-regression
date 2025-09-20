// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include <random>
#include <chrono>

namespace qsr {

thread_local std::mt19937 thread_local_rng(42);

}