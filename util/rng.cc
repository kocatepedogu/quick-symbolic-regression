// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include <random>

constexpr int random_seed = 42;

thread_local std::mt19937 thread_local_rng(random_seed);
