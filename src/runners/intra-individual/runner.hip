// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "runners/intra-individual/runner.hpp"
#include "runners/intra-individual/program/program.hpp"

#include "util/precision.hpp"
#include "vm/vm_debug.hpp"
#include "vm/vm_control.hpp"
#include "util/rng.hpp"

// Uncomment to enable buffer overflow checks
// #define CHECK_BUFFER_OVERFLOW

#ifdef CHECK_BUFFER_OVERFLOW
    #define DEBUGARGS ,\
        stack_length ,\
        intermediate_length
#else
    #define DEBUGARGS
#endif

using namespace qsr;
using namespace qsr::intra_individual;

constexpr int reduction_threads_per_block = 1024;

Runner::Runner(const int nweights, const bool use_cache, const HIPState *hipState) :
    GPUBaseRunner(nweights, hipState), use_cache(use_cache) {
    weights_d.resize(nweights);
}

__global__ static void
update_weights(Ptr1D<real> weight_grads, real *weight, real learning_rate)
{
    // --- 1. Initialization ---
    // Thread and block indices
    int tid = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    // Warp and lane identifiers
    constexpr int warpSize = 32;
    int warpId = tid / warpSize;
    int laneId = tid % warpSize;

    // Shared memory to store the partial sum of each warp.
    // The size is the number of warps per block.
    __shared__ real s_warp_sums[reduction_threads_per_block / warpSize];

    // Each thread loads its initial value from global memory into a register.
    real my_sum = (globalIdx < weight_grads.dim1) ? weight_grads[globalIdx] : (real)0.0f;

    // --- 2. Intra-Warp Reduction using Warp Shuffle ---
    // This loop sums values in registers within each warp without using shared memory.
    // The mask 0xFFFFFFFF ensures all threads in the warp participate.
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        my_sum += __shfl_down(0xFFFFFFFF, my_sum, offset);
    }
    // At this point, the first thread of each warp (laneId == 0) holds the sum for its warp.

    // --- 3. Store Warp Sums and Reduce Across Warps ---
    // The first thread (lane 0) of each warp writes its partial sum to shared memory.
    if (laneId == 0) {
        s_warp_sums[warpId] = my_sum;
    }

    // Synchronize to ensure all warp sums are written to shared memory before the final reduction.
    __syncthreads();

    // The first warp (warpId == 0) is now responsible for reducing the sums from all other warps.
    if (warpId == 0) {
        // The first warp's threads load the partial sums from shared memory.
        int num_warps_per_block = blockDim.x / warpSize;
        my_sum = (laneId < num_warps_per_block) ? s_warp_sums[laneId] : (real)0.0f;

        // Perform a final reduction within the first warp to get the total block sum.
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            my_sum += __shfl_down(0xFFFFFFFF, my_sum, offset);
        }

        // --- 4. Final Atomic Update ---
        // The first thread of the block (tid == 0) now holds the total sum
        // and performs the atomic update to global memory.
        if (laneId == 0) {
            atomicAdd(weight, -learning_rate * my_sum);
        }
    }
}

__global__ static void 
train(Ptr1D<Instruction> bytecode, 
      Ptr2D<real> X_d,
      Ptr1D<real> y_d,
      Ptr2D<real> stack_d,
      Ptr2D<real> intermediate_d,
      Ptr1D<real> weights_d,
      Ptr2D<real> weights_grad_d,
      Ptr1D<float> loss_d,
      bool enable_backpropagation,
      int stack_length, 
      int intermediate_length)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // Mask excess threads
    if (tid >= y_d.dim1) {
        return;
    }

    // Reset weight gradients that will be computed by this thread
    for (int j = 0; j < weights_d.dim1; ++j) {
        weights_grad_d[j,tid] = 0;
    }

    int program_counter = 0;
    int stack_pointer = 0;
    int intermediate_pointer = 0;

    const StackState s(stack_d, intermediate_d, stack_pointer, intermediate_pointer);
    const ControlState c(tid, tid, bytecode.dim1, bytecode, program_counter);
    const DataState d(X_d, y_d);
    const WeightState w(weights_d, weights_grad_d);

    // Forward propagate and evaluate loss
    vm_debug_print(tid, "Forward propagation");
    vm_control<FORWARD, INTRA_INDIVIDUAL, Ptr1D<real>>(c, d, s, w DEBUGARGS);

    // Print an empty line in between forward propagation output and backpropagation output
    vm_debug_print(tid, "");

    // Save squared difference divided by m [ (l/m)^2 * m = l^2/m ] as the loss
    const float squared_error = (float)stack_d[0,tid] * (float)stack_d[0,tid];
    loss_d[tid] = squared_error * (float)loss_d.dim1;

    // Backpropagate
    if (enable_backpropagation) {
        vm_debug_print(tid, "Backpropagation");
        vm_control<BACK, INTRA_INDIVIDUAL, Ptr1D<real>>(c, d, s, w DEBUGARGS);
    }
}

void Runner::run(const ProgramIndividual &p, std::shared_ptr<const Dataset> dataset, int epochs, real learning_rate) {
    for (int epoch = 0; epoch < epochs + 1; ++epoch) {
        bool enable_backpropagation = (epoch < epochs);

        // Run forward propagation and backpropagation to compute
        // weight gradients for each weight and for each data point
        launch_kernel(train, p.bytecode, dataset->X_d.ptr, dataset->y_d.ptr,
                      stack_d.ptr, intermediate_d.ptr,
                      weights_d.ptr, weights_grad_d.ptr, loss_d.ptr, 
                      enable_backpropagation,
                      p.stack_req, p.intermediate_req);

        // For every weight, sum gradient contributions from all data points using reduction
        // Apply gradient descent rule
        if (enable_backpropagation) {
            for (int i = 0; i < nweights; ++i) {
                launch_kernel(update_weights, weights_grad_d.ptr(i), &weights_d.ptr[i], learning_rate);
            }
        }
    }

    synchronize();
}

void Runner::calculate_kernel_dims(int m) {
    GPUBaseRunner::calculate_kernel_dims(m);

    int reduction_blocks_per_grid = (m + reduction_threads_per_block - 1) / 
        reduction_threads_per_block;

    reduction_grid_dim = dim3(reduction_blocks_per_grid);
    reduction_block_dim = dim3(reduction_threads_per_block);
}

void Runner::initialize_weights_and_losses(Expression &expression) {
    if (expression.weights.empty()) {
        for (int j = 0; j < nweights; ++j) {
            weights_d.ptr[j] = 2.0_r * (real)(thread_local_rng() % RAND_MAX) / (real)RAND_MAX - 1.0_r;
        }
    } 
    // If the expression already has weights, use them
    else {
        for (int j = 0; j < nweights; ++j) {
            weights_d.ptr[j] = expression.weights[j];
        }
    }
}

void Runner::save_weights_and_losses(Expression &expression) {
    // Compute total loss
    real total_loss = 0;
    #pragma omp simd reduction(+:total_loss)
    for (int i = 0; i < loss_d.ptr.dim1; ++i) {
        total_loss += loss_d.ptr[i];
    }

    // Save loss value to the original expression
    expression.loss = total_loss;

    // Save weights to the original expression
    expression.weights.resize(weights_d.ptr.dim1);
    for (int j = 0; j < weights_d.ptr.dim1; ++j) {
        expression.weights[j] = weights_d.ptr[j];
    }
}

void Runner::run(std::vector<Expression>& population, std::shared_ptr<const Dataset> dataset, int epochs, double learning_rate) {
    // Configure as many threads as data points
    calculate_kernel_dims(dataset->m);

    // Convert symbolic expressions to bytecode program
    Program program_pop(population);

    // Sequential loop over programs
    for (auto p : program_pop)
    {
        if (use_cache && population_cache.load(population[p.index])) {
            continue;
        }

        // Resize VM memory
        resize_arrays(p.stack_req, p.intermediate_req, dataset->m, dataset->m);

        // If the expression has no weights yet, initialize them randomly
        initialize_weights_and_losses(population[p.index]);

        // Run the epoch loop and train the program
        run(p, dataset, epochs, learning_rate);

        // Save weights and losses to the original expression
        save_weights_and_losses(population[p.index]);

        if (use_cache) {
            population_cache.save(population[p.index]);
        }
    }
}
