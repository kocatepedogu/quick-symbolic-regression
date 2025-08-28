// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef TOOLBOX_HPP
#define TOOLBOX_HPP

#include <memory>

#include "genetic/initialization/base.hpp"
#include "genetic/mutation/base.hpp"
#include "genetic/recombination/base.hpp"
#include "genetic/selection/base.hpp"

namespace qsr {
    /**
     * @brief The set of genetic operators used in a QuickSR model
     */

    class Toolbox {
    public:
        Toolbox(std::shared_ptr<BaseInitialization> initialization,
                std::shared_ptr<BaseMutation> mutation,
                std::shared_ptr<BaseRecombination> recombination,
                std::shared_ptr<BaseSelection> selection) noexcept;

        /// Population initializer shared by all islands
        const std::shared_ptr<BaseInitialization> initialization;

        /// Mutation operator shared by all islands
        const std::shared_ptr<BaseMutation> mutation;

        /// Crossover operator shared by all islands
        const std::shared_ptr<BaseRecombination> recombination;

        /// Selection operator shared by all islands
        const std::shared_ptr<BaseSelection> selection;
    };
}

#endif