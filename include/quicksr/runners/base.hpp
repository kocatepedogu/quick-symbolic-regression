// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef RUNNERS_BASE_HPP
#define RUNNERS_BASE_HPP

#include "expressions/expression.hpp"
#include "dataset/dataset.hpp"

#include <memory>
#include <vector>

namespace qsr {

class BaseRunner {
protected:
    const int nweights;

    Array1D<float> loss_d;

    Array2D<real> stack_d;

    Array2D<real> intermediate_d;

    Array2D<real> weights_grad_d;
    
    BaseRunner(int nweights);

    void resize_arrays(int stack_length, int intermediate_length, int nproblem, int nthreads);

public:
    virtual void run(std::vector<Expression>& population, std::shared_ptr<const Dataset> dataset, int epochs, double learning_rate);
};

}

#endif