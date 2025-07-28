#ifndef GENETIC_PROGRAMMING_ISLANDS_HPP
#define GENETIC_PROGRAMMING_ISLANDS_HPP

#include "../dataset/dataset.hpp"

#include "crossover/base.hpp"
#include "genetic_programming.hpp"
#include "mutation/base.hpp"
#include "selection/base.hpp"

class GeneticProgrammingIslands {
public:
    GeneticProgrammingIslands(const Dataset& dataset, 
                              int nweights, 
                              int npopulation, 
                              int max_initial_depth, 
                              int nislands, 
                              int ngenerations,
                              int nsupergenerations,
                              BaseMutation& mutation,
                              BaseCrossover& crossover,
                              BaseSelection& selection) noexcept;

    ~GeneticProgrammingIslands() noexcept;

    Expression iterate() noexcept;

private:
    /// Dataset shared by all islands
    const Dataset& dataset;

    /// Mutation operator shared by all islands
    BaseMutation &mutation;

    /// Crossover operator shared by all islands
    BaseCrossover &crossover;

    /// Selection operator shared by all islands
    BaseSelection &selection;

    /// Initialization parameters shared by all islands
    const int max_initial_depth;

    /// Number of iterations per supergeneration
    const int ngenerations;

    /// Number of supergenerations
    const int nsupergenerations;

    /// Properties of solutions
    const int nweights;

    /// Total population size
    const int npopulation;

    /// Array of islands
    GeneticProgramming **islands;

    /// Number of islands
    const int nislands;
};

#endif