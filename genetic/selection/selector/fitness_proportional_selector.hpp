#ifndef FITNESS_PROPORTIONAL_SELECTOR_HPP
#define FITNESS_PROPORTIONAL_SELECTOR_HPP

#include <vector>

#include "base.hpp"

class FitnessProportionalSelector : public BaseSelector {
public:
    constexpr FitnessProportionalSelector(int npopulation) : 
        npopulation(npopulation), probabilities(npopulation) {}

    void update(const Expression population[]);

    const Expression& select(const Expression population[]);

private:
    const int npopulation;

    std::vector<float> probabilities;
};

#endif