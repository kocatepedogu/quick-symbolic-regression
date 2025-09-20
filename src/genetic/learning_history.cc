// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "genetic/learning_history.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <bits/chrono.h>

namespace qsr {

/* Helper Functions */

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

static float loss_of_history_vector(const std::vector<float>& history) {
    if (history.empty()) {
        return std::numeric_limits<float>::max();
    } else {
        return history.back();
    }
}

static void add_to_history_vector(std::vector<float> &history, const Expression &expression) {
    float loss;

    if (history.empty()) {
        loss = expression.loss;
    } else {
        const float last_loss = history.back();
        loss = expression.loss < last_loss ? expression.loss : last_loss;
    }

    history.push_back(loss);
}

void LearningHistory::add_to_history(const Expression& expression) {
    add_to_history_vector(history_wrt_time, expression);
    add_to_history_vector(history_wrt_generation, expression);

    time.push_back(timestamp_in_milliseconds());
}

/* Member Functions */

LearningHistory LearningHistory::combine_with(const LearningHistory& other, const float previous_best_loss) const {
    LearningHistory combined;
    combine_with_wrt_time(combined, other, previous_best_loss);
    combine_with_wrt_generation(combined, other, previous_best_loss);
    return combined;
}

LearningHistory LearningHistory::concatenate_with(const LearningHistory& other) const {
    LearningHistory concatenated;
    concatenate_with_wrt_time(concatenated, other);
    concatenate_with_wrt_generation(concatenated, other);
    return concatenated;
}

void LearningHistory::combine_with_wrt_time(LearningHistory& result, const LearningHistory& other, const float previous_best_loss) const {
    // If self is empty, use the other's history wrt time
    if (this->history_wrt_time.empty()) {
        result.time = other.time;
        result.history_wrt_time = other.history_wrt_time;
    } else {
        // Ensure histories and times have the same size
        assert(this->history_wrt_time.size() == this->time.size());
        assert(other.history_wrt_time.size() == other.time.size());

        // Combine histories with respect to time
        int self_index = 0;
        int other_index = 0;
        float current_loss = previous_best_loss;
        while (self_index < this->time.size() || other_index < other.time.size())
        {
            bool take_self = self_index < this->time.size();
            if (take_self && other_index < other.time.size())
                take_self = this->time[self_index] < other.time[other_index];

            if (take_self)
            {
                const auto self_loss = this->history_wrt_time[self_index];
                current_loss = minimum_with_nan_check(self_loss, current_loss);
                result.history_wrt_time.emplace_back(current_loss);
                result.time.emplace_back(this->time[self_index++]);
            }
            else
            {
                const auto other_loss = other.history_wrt_time[other_index];
                current_loss = minimum_with_nan_check(other_loss, current_loss);
                result.history_wrt_time.emplace_back(current_loss);
                result.time.emplace_back(other.time[other_index++]);
            }
        }
    }
}

void LearningHistory::combine_with_wrt_generation(LearningHistory &result, const LearningHistory &other, const float previous_best_loss) const {
    // If self is empty use the other's history with respect to generation
    if (this->history_wrt_generation.empty()) {
        result.history_wrt_generation = other.history_wrt_generation;
    } else {
        // Ensure both histories have the same size
        assert(this->history_wrt_generation.size() == other.history_wrt_generation.size());

        // Combine histories with respect to generation
        float current_loss = previous_best_loss;
        for (const float self_loss : this->history_wrt_generation) {
            current_loss = minimum_with_nan_check(self_loss, current_loss);
            result.history_wrt_generation.emplace_back(current_loss);
        }
    }
}

void LearningHistory::concatenate_with_wrt_time(LearningHistory &result, const LearningHistory &other) const {
    // Ensure histories and times have the same size
    assert(this->history_wrt_time.size() == this->time.size());
    assert(other.history_wrt_time.size() == other.time.size());

    result.history_wrt_time = this->history_wrt_time;
    result.time = this->time;

    // Concatenate histories with respect to time
    float current_loss = loss_of_history_vector(this->history_wrt_time);
    for (int i = 0; i < other.time.size(); ++i) {
        const float other_loss = other.history_wrt_time[i];
        current_loss = minimum_with_nan_check(other_loss, current_loss);
        result.history_wrt_time.push_back(current_loss);
        result.time.push_back(other.time[i]);
    }
}

void LearningHistory::concatenate_with_wrt_generation(LearningHistory &result, const LearningHistory &other) const {
    result.history_wrt_generation = this->history_wrt_generation;

    // Concatenate histories with respect to generation
    float current_loss = loss_of_history_vector(this->history_wrt_generation);
    for (const float other_loss : other.history_wrt_generation) {
        current_loss = minimum_with_nan_check(other_loss, current_loss);
        result.history_wrt_generation.push_back(current_loss);
    }
}

}
