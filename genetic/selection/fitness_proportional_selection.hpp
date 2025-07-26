#ifndef FITNESS_PROPORTIONAL_SELECTION_HPP
#define FITNESS_PROPORTIONAL_SELECTION_HPP

#include <vector>

#include "../../expressions/expression.hpp"

class FitnessProportionalSelection {
public:
    constexpr FitnessProportionalSelection(int npopulation) :
        npopulation(npopulation), probabilities(npopulation) {}

    void initialize(const Expression population[]);

    const Expression& select(const Expression population[]);

private:
    const int npopulation;

    std::vector<float> probabilities;
};

#endif