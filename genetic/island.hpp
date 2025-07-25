#ifndef ISLAND_HPP
#define ISLAND_HPP

#include "../dataset/dataset.hpp"
#include "../expressions/expression.hpp"
#include "../inter-individual/runner.hpp"
#include "crossover.hpp"
#include "mutation.hpp"


class Island {
public:
    Island(const Dataset& dataset, 
           const int nweights, 
           const int npopulation, 
           const int max_initial_depth = 3, 
           const int max_mutation_depth = 3, 
           const float mutation_probability = 0.1,
           const float crossover_probability = 0.7) noexcept;

    void iterate(int niters) noexcept;

    void insert_solution(Expression e);

    Expression get_best_solution();

private:
    const Dataset& dataset;

    std::vector<Expression> population;

    Mutation mutator;
    Crossover crossover;

    inter_individual::Runner runner;

    const int nvars;
    const int nweights;
    const int npopulation;
};

#endif
