// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef RNG_HPP
#define RNG_HPP

#include <random>

namespace qsr {

extern thread_local std::mt19937 thread_local_rng;

}

#endif
