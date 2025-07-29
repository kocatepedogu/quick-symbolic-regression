#ifndef GENETIC_PROGRAMMING_ISLANDS_HPP
#define GENETIC_PROGRAMMING_ISLANDS_HPP

#include "../dataset/dataset.hpp"

#include "crossover/base.hpp"
#include "genetic_programming.hpp"
#include "initializer/base.hpp"
#include "mutation/base.hpp"
#include "selection/base.hpp"

class GeneticProgrammingIslands {
public:
    GeneticProgrammingIslands(const Dataset& dataset, 
                              int nweights, 
                              int npopulation, 
                              int nislands, 
                              int ngenerations,
                              int nsupergenerations,
                              BaseInitializer& initializer,
                              BaseMutation& mutation,
                              BaseCrossover& crossover,
                              BaseSelection& selection) noexcept;

    ~GeneticProgrammingIslands() noexcept;

    Expression iterate() noexcept;

private:
    /// Dataset shared by all islands
    const Dataset& dataset;

    /// Population initializer shared by all islands
    BaseInitializer &initializer;

    /// Mutation operator shared by all islands
    BaseMutation &mutation;

    /// Crossover operator shared by all islands
    BaseCrossover &crossover;

    /// Selection operator shared by all islands
    BaseSelection &selection;

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