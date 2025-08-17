// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef CONFIG_HPP
#define CONFIG_HPP

#include "function_set.hpp"
#include <memory>

namespace qsr {
    struct Config {
        Config();
        
        Config(int nvars, int nweights, int max_depth, int npopulation, int noffspring, std::shared_ptr<FunctionSet> function_set);

        int nvars;
        int nweights;
        int max_depth;
        int npopulation;
        int noffspring;

        std::shared_ptr<FunctionSet> function_set;
    };
}

#endif // CONFIG_HPP