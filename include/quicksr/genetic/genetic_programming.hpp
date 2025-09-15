// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef GENETIC_PROGRAMMING_HPP
#define GENETIC_PROGRAMMING_HPP

#include "dataset/dataset.hpp"
#include "expressions/expression.hpp"
#include "runners/base.hpp"

#include "genetic/common/function_set.hpp"
#include "genetic/common/toolbox.hpp"
#include "genetic/selection/selector/base.hpp"

#include "learning_history.hpp"

namespace qsr {

class GeneticProgramming {
public:
    GeneticProgramming(const Config &config, const Toolbox &toolbox, std::shared_ptr<BaseRunner> runner) noexcept;

    LearningHistory fit(std::shared_ptr<const Dataset> dataset, int ngenerations, int nepochs, float learning_rate) noexcept;

    void insert_solution(Expression e);

    Expression *get_best_solution();
    Expression *get_worst_solution();

    inline auto& get_population() { return population; }

private:
    std::vector<Expression> population;

    std::shared_ptr<BaseMutator> mutator;
    std::shared_ptr<BaseRecombiner> recombiner;
    std::shared_ptr<BaseSelector> selector;
    std::shared_ptr<BaseRunner> runner;
    std::shared_ptr<FunctionSet> function_set;

    Config config;

    bool initialized;
};

}

#endif
