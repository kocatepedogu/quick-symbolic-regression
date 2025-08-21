// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "../base.hpp"

#include "../../expressions/expression.hpp"
#include "../../dataset/dataset.hpp"

#include <memory>

namespace qsr::cpu {
    class Runner : public BaseRunner {
    private:
        const int nweights;

    public:
        Runner(int nweights);

        void run(std::vector<Expression>& population, std::shared_ptr<const Dataset> dataset, int epochs, float learning_rate) override;
    };
}