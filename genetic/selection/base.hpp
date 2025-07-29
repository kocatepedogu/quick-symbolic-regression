#ifndef SELECTION_BASE_HPP
#define SELECTION_BASE_HPP

#include "selector/base.hpp"

#include <memory>

class BaseSelection {
public:
    virtual std::shared_ptr<BaseSelector> get_selector(int npopulation);
};

#endif