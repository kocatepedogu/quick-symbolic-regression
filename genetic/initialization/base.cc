// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "base.hpp"

namespace qsr {
    std::shared_ptr<BaseInitializer> BaseInitialization::get_initializer(const Config &config) {
        fprintf(stderr, "Unimplemented base method called.");
        abort();
    }
}