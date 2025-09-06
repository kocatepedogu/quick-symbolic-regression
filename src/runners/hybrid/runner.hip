// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "runners/hybrid/runner.hpp"

#include "util/rng.hpp"
#include "vm/vm_types.hpp"
#include "vm/vm_debug.hpp"
#include "vm/vm_control.hpp"
#include "runners/inter-individual/program/program.hpp"

// Uncomment to enable buffer overflow checks
// #define CHECK_BUFFER_OVERFLOW

#ifdef CHECK_BUFFER_OVERFLOW
    #define DEBUGARGS ,\
        stack_req ,\
        intermediate_req
#else
    #define DEBUGARGS
#endif

using namespace qsr;
using namespace qsr::hybrid;

Runner::Runner(int nweights) : 
    GPUBaseRunner(nweights) {}

static inline __device__ void 
reset_gradients_and_losses(int program_idx, Ptr2D<float> &weights_grad_d, Ptr1D<float> &loss_d) 
{
    // Reset weight gradients
    for (int weight_idx = 0; weight_idx < weights_grad_d.dim1; ++weight_idx) {
        weights_grad_d[weight_idx][program_idx] = 0;
    }

    // Reset total loss
    loss_d[program_idx] = 0;
}

static inline __device__ void 
update_weights(int program_idx, float learning_rate, Ptr2D<float> &weights_d, Ptr2D<float> &weights_grad_d) 
{
    // For each weight
    for (int weight_idx = 0; weight_idx < weights_d.dim1; ++weight_idx) {
        // Apply gradient descent
        weights_d[weight_idx, program_idx] -= learning_rate * weights_grad_d[weight_idx][program_idx];
    }
}

__global__ static void 
train(Ptr2D<Instruction> bytecode, Ptr2D<float> X_d, Ptr1D<float> y_d, 
      Ptr2D<float> stack_d, Ptr2D<float> intermediate_d,
      Ptr2D<float> weights_d, Ptr2D<float> weights_grad_d, Ptr1D<float> loss_d,
      const int nepochs, const float learning_rate,
      int stack_req, int intermediate_req) 
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    // Find program index and datapoint index
    const int datapoint_block_dim = 32;
    const int program_idx = tid / datapoint_block_dim;
    const int datapoint_block_idx = tid % datapoint_block_dim;

    // Mask excess threads
    if (program_idx >= bytecode.dim2 || datapoint_block_idx >= datapoint_block_dim) {
        return;
    }

    for (int epoch = 0; epoch < nepochs + 1; ++epoch) {
        // At the beginning of each epoch, reset gradients and losses
        if (datapoint_block_idx == 0) {
            reset_gradients_and_losses(program_idx, weights_grad_d, loss_d);
        }

        // For all datapoints, compute and sum weight gradients
        for (int datapoint_idx = datapoint_block_idx; datapoint_idx < y_d.dim1; datapoint_idx += datapoint_block_dim) {
            int program_counter = 0;
            int stack_pointer = 0;
            int intermediate_pointer = 0;

            const StackState s(stack_d, intermediate_d, stack_pointer, intermediate_pointer);
            const ControlState c(tid, datapoint_idx, bytecode.dim1, bytecode, program_counter);
            const DataState d(X_d, y_d);
            const WeightState w(weights_d, weights_grad_d);

            // Forward propagate and evaluate loss
            vm_debug_print(tid, "Forward propagation");
            vm_control<FORWARD, HYBRID, Ptr2D<float>>(c, d, s,w DEBUGARGS);

            // Print an empty line in between forward propagation output and backpropagation output
            vm_debug_print(tid, "");

            // Add squared difference divided by m [ (l/m)^2 * m = l^2/m ] to total loss
            if (epoch == nepochs) {
                atomicAdd(&loss_d[program_idx], powf(stack_d[0,tid], 2) * (float)y_d.dim1);
            }

            // Backpropagate
            if (epoch < nepochs) {
                vm_debug_print(tid, "Backpropagation");
                vm_control<BACK, HYBRID, Ptr2D<float>>(c, d, s, w DEBUGARGS);
            }
        }

        // At the end of each epoch, update weights
        if (datapoint_block_idx == 0 && epoch < nepochs) { 
            update_weights(program_idx, learning_rate, weights_d, weights_grad_d); 
        }
    }
}

void Runner::initialize_weights_and_losses(std::vector<Expression>& population) {
    // Initialize weights array
    for (int i = 0; i < nweights; ++i) {
        for (int j = 0; j < population.size(); ++j) {
            const auto &expression = population[j];

            // If the expression has no weights yet, initialize them randomly
            if (expression.weights.empty()) {
                weights_d.ptr[i,j] = 2 * (thread_local_rng() % RAND_MAX) / (float)RAND_MAX - 1;
            } 
            // If the expression already has weights, use them
            else {
                weights_d.ptr[i,j] = expression.weights[i];
            }   
        }
    }

    // Initialize loss array with NaN's
    for (int j = 0; j < population.size(); ++j) {
        loss_d.ptr[j] = std::numeric_limits<float>::quiet_NaN();
    }
}

void Runner::save_weights_and_losses(std::vector<Expression>& population) {
    // Write loss values to the original expressions
    for (int i = 0; i < population.size(); ++i) {
        population[i].loss = loss_d.ptr[i];
    }

    // Write weight values to the original expressions
    for (int i = 0; i < population.size(); ++i) {
        population[i].weights.resize(nweights);
        for (int j = 0; j < nweights; ++j) {
            population[i].weights[j] = weights_d.ptr[j,i];
        }
    }
}

void Runner::run(std::vector<Expression>& population, std::shared_ptr<const Dataset> dataset, int nepochs, float learning_rate) {
    // Convert symbolic expressions to bytecode program
    inter_individual::Program program(population);

    // Every warp/wave executes a single expression, which takes 32 threads per expression
    calculate_kernel_dims(program.num_of_individuals * 32);

    // Resize arrays
    weights_d.resize(nweights, population.size());
    resize_arrays(program.stack_req, program.intermediate_req, population.size(), nthreads);

    // Initialize weights and losses
    initialize_weights_and_losses(population);

    // Run virtual machine
    launch_kernel(
        train,
        program.bytecode.ptr, 
        dataset->X_d.ptr, dataset->y_d.ptr,
        stack_d.ptr, intermediate_d.ptr,
        weights_d.ptr, weights_grad_d.ptr, loss_d.ptr,
        nepochs, learning_rate,
        program.stack_req, program.intermediate_req
    );

    // Wait for the kernel to finish
    synchronize();

    // Save weights and losses to the original expressions
    save_weights_and_losses(population);
}
