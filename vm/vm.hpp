// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef VM_HPP
#define VM_HPP

#include "../compiler/bytecode.hpp"
#include "../dataset/dataset.hpp"

void forward_propagate(const Program& program, const Dataset& dataset);

#endif