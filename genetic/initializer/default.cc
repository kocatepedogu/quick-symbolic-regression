#include "default.hpp"

void DefaultInitializer::initialize(std::vector<Expression>& population) {
    population.clear();
    for (int i = 0; i < npopulation; ++i) {
        population.push_back(generator.generate());
    }
}