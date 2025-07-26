#ifndef GENETIC_PROGRAMMING_HPP
#define GENETIC_PROGRAMMING_HPP

#include "../dataset/dataset.hpp"
#include "../expressions/expression.hpp"
#include "../inter-individual/runner.hpp"
#include "crossover/default.hpp"
#include "mutation/default.hpp"


class Island {
public:
    Island(const Dataset& dataset, 
           const int nweights, 
           const int npopulation, 
           const int max_initial_depth = 3, 
           const int max_mutation_depth = 3, 
           const float mutation_probability = 0.7,
           const float crossover_probability = 1.0) noexcept;

    void iterate(int niters) noexcept;

    void insert_solution(Expression e);

    Expression get_best_solution();

private:
    const Dataset& dataset;

    std::vector<Expression> population;
    std::vector<float> probabilities;

    Mutation mutator;
    Crossover crossover;

    void parent_selection_fitness_proportional_probs() noexcept;
    int parent_selection_fitness_proportional() const noexcept;

    inter_individual::Runner runner;

    const int nvars;
    const int nweights;
    const int npopulation;
};

#endif
