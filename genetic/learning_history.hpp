#ifndef LEARNING_RESULT_HPP
#define LEARNING_RESULT_HPP

#include <vector>

#include "../expressions/expression.hpp"

class LearningHistory {
public:
    // Adds a fitness value to the learning history
    void add_to_history(const Expression& expression);

    // Combines two LearningHistory objects
    // For each iteration, take the best fitness value from either of the histories
    LearningHistory combine_with(LearningHistory other);

    // Concatenates two LearningHistory objects
    // Append the other's history to this one
    LearningHistory concatenate_with(LearningHistory other);

    // Returns the learning history
    constexpr const std::vector<float>& get_learning_history() const {
        return history;
    }

private:
    std::vector<float> history;
};

#endif