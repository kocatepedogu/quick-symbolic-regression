#ifndef GENETIC_PROGRAMMING_HPP
#define GENETIC_PROGRAMMING_HPP

#include "../dataset/dataset.hpp"
#include "../expressions/expression.hpp"
#include "../runners/base.hpp"

#include "crossover/base.hpp"
#include "initializer/base.hpp"
#include "mutation/base.hpp"
#include "selection/base.hpp"
#include "selection/selector/base.hpp"

class GeneticProgramming {
public:
    GeneticProgramming(std::shared_ptr<const Dataset> dataset, 
                       int nweights, 
                       int npopulation, 
                       std::shared_ptr<BaseInitializer> initializer,
                       std::shared_ptr<BaseMutation> mutator,
                       std::shared_ptr<BaseCrossover> crossover,
                       std::shared_ptr<BaseSelection> selection,
                       std::shared_ptr<BaseRunner> runner) noexcept;

    void fit(int ngenerations, int nepochs, float learning_rate) noexcept;

    void insert_solution(Expression e);

    Expression get_best_solution();

private:
    std::shared_ptr<const Dataset> dataset;

    std::vector<Expression> population;

    std::shared_ptr<BaseInitializer> initializer;
    std::shared_ptr<BaseMutation> mutator;
    std::shared_ptr<BaseCrossover> crossover;
    std::shared_ptr<BaseSelection> selection;
    std::shared_ptr<BaseSelector> selector;
    std::shared_ptr<BaseRunner> runner;

    const int nvars;
    const int nweights;
    const int npopulation;
};

#endif
