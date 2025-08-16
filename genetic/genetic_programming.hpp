// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef GENETIC_PROGRAMMING_HPP
#define GENETIC_PROGRAMMING_HPP

#include "../dataset/dataset.hpp"
#include "../expressions/expression.hpp"
#include "../runners/base.hpp"

#include "common/function_set.hpp"
#include "recombination/base.hpp"
#include "initialization/base.hpp"
#include "mutation/base.hpp"
#include "selection/base.hpp"
#include "selection/selector/base.hpp"

#include "learning_history.hpp"

namespace qsr {

class GeneticProgramming {
public:
    GeneticProgramming(std::shared_ptr<const Dataset> dataset, 
                       int nweights, 
                       int npopulation, 
                       int noffspring,
                       int max_depth,
                       std::shared_ptr<BaseInitialization> initializer,
                       std::shared_ptr<BaseMutation> mutator,
                       std::shared_ptr<BaseRecombination> recombination,
                       std::shared_ptr<BaseSelection> selection,
                       std::shared_ptr<BaseRunner> runner,
                       std::shared_ptr<FunctionSet> function_set) noexcept;

    LearningHistory fit(int ngenerations, int nepochs, float learning_rate) noexcept;

    void insert_solution(Expression e);

    Expression *get_best_solution();
    Expression *get_worst_solution();

private:
    std::shared_ptr<const Dataset> dataset;

    std::vector<Expression> population;

    std::shared_ptr<BaseInitialization> initialization;
    std::shared_ptr<BaseMutation> mutation;
    std::shared_ptr<BaseRecombination> recombination;
    std::shared_ptr<BaseSelection> selection;

    std::shared_ptr<BaseInitializer> initializer;
    std::shared_ptr<BaseMutator> mutator;
    std::shared_ptr<BaseRecombiner> recombiner;
    std::shared_ptr<BaseSelector> selector;

    std::shared_ptr<BaseRunner> runner;
    std::shared_ptr<FunctionSet> function_set;

    const int nvars;
    const int nweights;
    const int npopulation;
    const int noffspring;
    const int max_depth;

    bool initialized;
};

}

#endif
