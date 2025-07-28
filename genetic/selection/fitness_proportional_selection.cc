#include "fitness_proportional_selection.hpp"
#include "selector/fitness_proportional_selector.hpp"

BaseSelector *FitnessProportionalSelection::get_selector() {
    return new FitnessProportionalSelector(npopulation);
}
