#include "fitness_proportional_selection.hpp"
#include "selector/fitness_proportional_selector.hpp"

std::shared_ptr<BaseSelector> FitnessProportionalSelection::get_selector(int npopulation) noexcept {
    return std::make_shared<FitnessProportionalSelector>(npopulation);
}
