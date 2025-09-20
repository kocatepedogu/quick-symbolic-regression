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
     * The loss of the given expression is appended to the history.
     * The value is capped by the previously added loss.
     */ 
    void add_to_history(const Expression& expression);

    /**
     * @brief Combines two LearningHistory objects
     * @param other The other LearningHistory object to combine with
     * @param previous_best_loss Best loss observed prior to the current supergeneration
     *
     * @details
     * For every iteration, takes the best fitness value from either of the histories.
     */
    [[nodiscard]] LearningHistory combine_with(const LearningHistory& other, float previous_best_loss) const;

    /**
     * @brief Concatenates two LearningHistory objects
     * @param other The other LearningHistory object to concatenate with
     *
     * @details
     * Appends the other's history to the end of this one.
     */
    [[nodiscard]] LearningHistory concatenate_with(const LearningHistory& other) const;

    /**
     * @brief Returns the evolution of loss with respect to time
     * @return A constant reference to the vector of loss values
     */
    [[nodiscard]] const std::vector<float>& get_learning_history_wrt_time() const {
        return history_wrt_time;
    }

    /**
     * @brief Returns the evolution of loss with respect to generation
     * @return A constant reference to the vector of loss values
     */
    [[nodiscard]] const std::vector<float>& get_learning_history_wrt_generation() const {
        return history_wrt_generation;
    }

    /**
     * @brief Returns the timestamps of loss values provided by get_learning_history_wrt_time
     * @return A constant reference to the vector of time values
     */
    [[nodiscard]] const std::vector<long>& get_time_history() const {
        return time;
    }

private:
    void combine_with_wrt_time(LearningHistory& result, const LearningHistory& other, float previous_best_loss) const;

    void combine_with_wrt_generation(LearningHistory& result, const LearningHistory& other, float previous_best_loss) const;

    void concatenate_with_wrt_time(LearningHistory& result, const LearningHistory& other) const;

    void concatenate_with_wrt_generation(LearningHistory& result, const LearningHistory& other) const;

    std::vector<float> history_wrt_time;

    std::vector<float> history_wrt_generation;

    std::vector<long> time;
};

}

#endif