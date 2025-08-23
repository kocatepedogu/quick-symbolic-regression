// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "../base.hpp"

#include "../intra-individual/program/program.hpp"
#include "../../expressions/expression.hpp"
#include "../../dataset/dataset.hpp"

#include <memory>

namespace qsr::cpu {
    class Runner : public BaseRunner {
    private:
        Array1D<float> weights_d;

        void reset_gradients_and_losses();

        void update_weights(float learning_rate);

        void train(const intra_individual::ProgramIndividual &p, std::shared_ptr<const Dataset> dataset, int epochs, float learning_rate);

        void initialize_weights(Expression& expression);

        void save_weights_and_losses(Expression& expression);

    public:
        Runner(int nweights);

        void run(std::vector<Expression>& population, std::shared_ptr<const Dataset> dataset, int epochs, float learning_rate) override;
    };
}