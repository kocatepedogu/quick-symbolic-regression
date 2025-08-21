// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef RUNNERS_BASE_HPP
#define RUNNERS_BASE_HPP

#include "../expressions/expression.hpp"
#include "../dataset/dataset.hpp"

#include <memory>
#include <vector>

namespace qsr {

class BaseRunner {
public:
    virtual void run(std::vector<Expression>& population, std::shared_ptr<const Dataset> dataset, int epochs, float learning_rate);
};

}

#endif