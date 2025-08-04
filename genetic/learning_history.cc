// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "learning_history.hpp"

#include <cassert>
#include <cmath>

void LearningHistory::add_to_history(const Expression& expression) {
    history.push_back(expression.loss);
}

LearningHistory LearningHistory::combine_with(LearningHistory other) {
    // If self is empty, use the other's history
    if (this->history.empty()) {
        return other;
    }

    LearningHistory combined;

    // Ensure both histories have the same size
    assert(this->history.size() == other.history.size());

    for (int i = 0; i < this->history.size(); ++i) {
        // If both are NaN
        if (std::isnan(this->history[i]) && std::isnan(other.history[i])) {
            combined.history.push_back(std::numeric_limits<float>::quiet_NaN());
        }
        // If this is not NaN, but other is NaN
        else if (!std::isnan(this->history[i]) && std::isnan(other.history[i])) {
            combined.history.push_back(this->history[i]);
        }
        // If other is not NaN, but this is NaN
        else if (std::isnan(this->history[i]) && !std::isnan(other.history[i])) {
            combined.history.push_back(other.history[i]);
        }
        // If neither is NaN
        else {
             // If this has lower loss, use this
            if (this->history[i] < other.history[i]) {
                combined.history.push_back(this->history[i]);
            } 
            // If other has lower loss, use other
            else {
                combined.history.push_back(other.history[i]);
            }
        }
    }

    return combined;
}

LearningHistory LearningHistory::concatenate_with(LearningHistory other) {
    LearningHistory concatenated;

    concatenated.history = this->history;
    concatenated.history.insert(concatenated.history.end(), other.history.begin(), other.history.end());

    return concatenated;
}