// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include <random>
#include <chrono>

namespace qsr {

static thread_local unsigned random_seed = static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count());

thread_local std::mt19937 thread_local_rng(random_seed);

}