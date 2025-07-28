#ifndef FITNESS_PROPORTIONAL_SELECTION_HPP
#define FITNESS_PROPORTIONAL_SELECTION_HPP

#include "base.hpp"
#include "selector/base.hpp"

class FitnessProportionalSelection : public BaseSelection {
public:
    BaseSelector *get_selector(int npopulation) noexcept;
};

#endif