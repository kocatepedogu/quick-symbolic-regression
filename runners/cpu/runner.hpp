// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "../base.hpp"

#include "../../../compiler/ir.hpp"
#include "../../expressions/expression.hpp"
#include "../../dataset/dataset.hpp"

#include <memory>

namespace qsr::cpu {
    class Runner : public BaseRunner {
    private:
        Array1D<float> weights_d;

        float loss;

        void reset_gradients_and_losses();

        void update_weights(float learning_rate);

        void train(Instruction *bytecode, int num_of_instructions, std::shared_ptr<const Dataset> dataset, int epochs, float learning_rate,
                    int stack_req, int intermediate_req);

        void initialize_weights(Expression& expression);

        void save_weights_and_losses(Expression& expression);

    public:
        Runner(int nweights);

        void run(std::vector<Expression>& population, std::shared_ptr<const Dataset> dataset, int epochs, float learning_rate) override;
    };
}