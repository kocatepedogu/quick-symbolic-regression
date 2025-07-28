#ifndef GENETIC_PROGRAMMING_HPP
#define GENETIC_PROGRAMMING_HPP

#include "../dataset/dataset.hpp"
#include "../expressions/expression.hpp"
#include "../inter-individual/runner.hpp"

#include "crossover/base.hpp"
#include "mutation/base.hpp"
#include "selection/base.hpp"


class GeneticProgramming {
public:
    GeneticProgramming(const Dataset& dataset, 
                       int nweights, 
                       int npopulation, 
                       int max_initial_depth, 
                       BaseMutation& mutator,
                       BaseCrossover& crossover,
                       BaseSelection& selection) noexcept;

    void iterate(int niters) noexcept;

    void insert_solution(Expression e);

    Expression get_best_solution();

private:
    const Dataset& dataset;

    std::vector<Expression> population;

    BaseMutation &mutator;
    BaseCrossover &crossover;
    BaseSelection &selection;

    inter_individual::Runner runner;

    const int nvars;
    const int nweights;
    const int npopulation;
};

#endif
