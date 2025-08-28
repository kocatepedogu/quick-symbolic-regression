// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef RUNNER_GENERATOR_BASE_HPP
#define RUNNER_GENERATOR_BASE_HPP

#include "runners/base.hpp"

#include <memory>

namespace qsr {

class BaseRunnerGenerator {
public:
    virtual std::shared_ptr<BaseRunner> generate(int nweights);
};

}

#endif