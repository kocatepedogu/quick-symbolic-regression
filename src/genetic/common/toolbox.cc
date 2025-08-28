// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "genetic/common/toolbox.hpp"

namespace qsr {

Toolbox::Toolbox(std::shared_ptr<BaseInitialization> initialization,
                 std::shared_ptr<BaseMutation> mutation,
                 std::shared_ptr<BaseRecombination> recombination,
                 std::shared_ptr<BaseSelection> selection) noexcept :
                 initialization(initialization),
                 mutation(mutation),
                 recombination(recombination),
                 selection(selection) {}
                 
}