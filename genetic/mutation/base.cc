// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "base.hpp"
#include <cstdio>

Expression BaseMutation::mutate(const Expression &expr) noexcept {
    fprintf(stderr, "Unimplemented base method called.");
    abort();
}