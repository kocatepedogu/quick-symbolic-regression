// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef LEARNING_RESULT_HPP
#define LEARNING_RESULT_HPP

#include <vector>

#include "expressions/expression.hpp"

namespace qsr {

class LearningHistory {
public:
    /**
     * @brief Adds a loss value to the learning history
     * @param expression The expression whose loss value is to be added
     *
     * @details
     * The loss of the given expression is simply appended to the history.
     *
     * Note that the new value can be higher than the previous one if the algorithm
     * allows for increase in loss over time.
     */ 
    void add_to_history(const Expression& expression);

    /**
     * @brief Combines two LearningHistory objects
     * @param other The other LearningHistory object to combine with
     *
     * @details
     * For every iteration, takes the best fitness value from either of the histories.
     */
    LearningHistory combine_with(LearningHistory other);

    /**
     * @brief Concatenates two LearningHistory objects
     * @param other The other LearningHistory object to concatenate with
     *
     * @details
     * Appends the other's history to the end of this one.
     */
    LearningHistory concatenate_with(LearningHistory other) const;

    /**
     * @brief Returns the learning history
     *
     * @return A constant reference to the vector of fitness values
     */
    constexpr const std::vector<float>& get_learning_history() const {
        return history;
    }

    constexpr const std::vector<long>& get_time_history() const {
        return time;
    }

private:
    std::vector<float> history;

    std::vector<long> time;
};

}

#endif