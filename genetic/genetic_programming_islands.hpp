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
    GeneticProgrammingIslands(std::shared_ptr<Dataset> dataset, 
                              int nislands, 
                              int nweights, 
                              int npopulation, 
                              int ngenerations,
                              int nsupergenerations,
                              std::shared_ptr<BaseInitializer> initializer,
                              std::shared_ptr<BaseMutation> mutation,
                              std::shared_ptr<BaseCrossover> crossover,
                              std::shared_ptr<BaseSelection> selection) noexcept;

    ~GeneticProgrammingIslands() noexcept;

    std::string iterate() noexcept;

private:
    /// Dataset shared by all islands
    std::shared_ptr<Dataset> dataset;

    /// Population initializer shared by all islands
    std::shared_ptr<BaseInitializer> initializer;

    /// Mutation operator shared by all islands
    std::shared_ptr<BaseMutation> mutation;

    /// Crossover operator shared by all islands
    std::shared_ptr<BaseCrossover> crossover;

    /// Selection operator shared by all islands
    std::shared_ptr<BaseSelection> selection;

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