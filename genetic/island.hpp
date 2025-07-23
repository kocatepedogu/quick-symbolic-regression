#ifndef ISLAND_HPP
#define ISLAND_HPP

#include "../dataset/dataset.hpp"
#include "../expressions/expression.hpp"
#include "../inter-individual/runner.hpp"
#include "mutation.hpp"


class Island {
public:
    Island(const Dataset& dataset, 
           const int nweights, 
           const int npopulation, 
           const int max_initial_depth = 3, 
           const int max_mutation_depth = 3, 
           const float mutation_probability = 0.5) noexcept;

    void iterate(int niters) noexcept;

private:
    const Dataset& dataset;

    std::vector<Expression> population;

    Mutation mutator;

    inter_individual::Runner runner;

    const int nvars;
    const int nweights;
    const int npopulation;
};

#endif
