// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef INITIALIZATION_BASE_HPP
#define INITIALIZATION_BASE_HPP

#include "initializer/base.hpp"

#include "../common/config.hpp"

#include <memory>
namespace qsr {
    class BaseInitialization {
    public:
        virtual std::shared_ptr<BaseInitializer> get_initializer(const Config &config);
    };
}

#endif