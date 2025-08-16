// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef FUNCTION_SET_HPP
#define FUNCTION_SET_HPP

#include <vector>
#include <string>
#include <algorithm>

namespace qsr {
    struct FunctionSet {
        constexpr FunctionSet(std::vector<std::string> functions) {
            addition = std::find(functions.begin(), functions.end(), "+") != functions.end();
            subtraction = std::find(functions.begin(), functions.end(), "-") != functions.end();
            multiplication = std::find(functions.begin(), functions.end(), "*") != functions.end();
            division = std::find(functions.begin(), functions.end(), "/") != functions.end();
            sine = std::find(functions.begin(), functions.end(), "sin") != functions.end();
            cosine = std::find(functions.begin(), functions.end(), "cos") != functions.end();
            exponential = std::find(functions.begin(), functions.end(), "exp") != functions.end();
            rectified_linear_unit = std::find(functions.begin(), functions.end(), "relu") != functions.end();
        }

        bool addition;
        bool subtraction;
        bool multiplication;
        bool division;
        bool sine;
        bool cosine;
        bool exponential;
        bool rectified_linear_unit;
    };
}

#endif