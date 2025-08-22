// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "../base.hpp"

#include "../../../compiler/ir.hpp"
#include "../../expressions/expression.hpp"
#include "../../dataset/dataset.hpp"

#include <memory>

namespace qsr::cpu {
    constexpr int initial_array_size = 128;

    class Runner : public BaseRunner {
    private:
        const int nweights;

        Array2D<float> stack_d;
        Array2D<float> intermediate_d;
        Array1D<float> weights_d;
        Array2D<float> weights_grad_d;

        void initialize_weights(Expression& expression);

        void update_weights(float learning_rate);

        void reset_gradients();

        float train(Instruction *bytecode, int num_of_instructions, std::shared_ptr<const Dataset> dataset, int epochs, float learning_rate);

    public:
        Runner(int nweights);

        void run(std::vector<Expression>& population, std::shared_ptr<const Dataset> dataset, int epochs, float learning_rate) override;
    };
}