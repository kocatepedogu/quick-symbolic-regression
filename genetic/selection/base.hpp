#ifndef SELECTION_BASE_HPP
#define SELECTION_BASE_HPP

#include "selector/base.hpp"

class BaseSelection {
public:
    virtual BaseSelector *get_selector(int npopulation) = 0;
};

#endif