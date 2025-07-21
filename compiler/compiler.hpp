// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef COMPILER_HPP
#define COMPILER_HPP

#include "../expressions/expression.hpp"
#include "programpopulation.hpp"

ProgramPopulation compile(const std::vector<Expression>& exp_pop) noexcept;

#endif