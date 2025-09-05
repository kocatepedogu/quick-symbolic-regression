// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "genetic/learning_history.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <bits/chrono.h>

namespace qsr {

static long timestamp_in_milliseconds() {
    const auto now = std::chrono::system_clock::now();
    const auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
}

static float minimum_with_nan_check(const float a, const float b)
{
    // If both are NaN
    if (std::isnan(a) && std::isnan(b)) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    // If this is not NaN, but other is NaN
    else if (!std::isnan(a) && std::isnan(b)) {
        return a;
    }
    // If other is not NaN, but this is NaN
    else if (std::isnan(a) && !std::isnan(b)) {
        return b;
    }
    // If neither is NaN
    else {
        return std::min(a, b);
    }
}

void LearningHistory::add_to_history(const Expression& expression) {
    history.push_back(expression.loss);
    time.push_back(timestamp_in_milliseconds());
}

LearningHistory LearningHistory::combine_with(LearningHistory other) {
    // If self is empty, use the other's history
    if (this->history.empty()) {
        return other;
    }

    LearningHistory combined;

    // Ensure histories and times have the same size
    assert(this->history.size() == this->time.size());
    assert(other.history.size() == other.time.size());

    int self_index = 0;
    int other_index = 0;

    float current_loss = std::numeric_limits<float>::max();
    while (self_index < this->time.size() || other_index < other.time.size())
    {
        bool take_self = self_index < this->time.size();
        if (take_self && other_index < other.time.size())
            take_self = this->time[self_index] < other.time[other_index];

        if (take_self)
        {
            const auto self_loss = this->history[self_index];
            current_loss = minimum_with_nan_check(self_loss, current_loss);
            combined.history.emplace_back(current_loss);
            combined.time.emplace_back(this->time[self_index++]);
        }
        else
        {
            const auto other_loss = other.history[other_index];
            current_loss = minimum_with_nan_check(other_loss, current_loss);
            combined.history.emplace_back(current_loss);
            combined.time.emplace_back(other.time[other_index++]);
        }
    }

    return combined;
}

LearningHistory LearningHistory::concatenate_with(LearningHistory other) const
{
    LearningHistory concatenated;

    concatenated.history = this->history;
    concatenated.time = this->time;

    concatenated.history.insert(concatenated.history.end(), other.history.begin(), other.history.end());
    concatenated.time.insert(concatenated.time.end(), other.time.begin(), other.time.end());

    return concatenated;
}

}
