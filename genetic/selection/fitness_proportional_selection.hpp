#ifndef FITNESS_PROPORTIONAL_SELECTION_HPP
#define FITNESS_PROPORTIONAL_SELECTION_HPP

#include "base.hpp"
#include "selector/base.hpp"

class FitnessProportionalSelection : public BaseSelection {
public:
    constexpr FitnessProportionalSelection(int npopulation) :
        npopulation(npopulation) {}

    BaseSelector *get_selector();

private:
    const int npopulation;
};

#endif