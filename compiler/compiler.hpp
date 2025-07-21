// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef COMPILER_HPP
#define COMPILER_HPP

#include "../expressions/expression.hpp"
#include "program.hpp"

Program compile(const Expression& e) noexcept;

#endif