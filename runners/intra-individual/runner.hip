#include "runner.hpp"

#include "./program/program.hpp"

#include "../../vm/vm_debug.hpp"
#include "../../vm/vm_control.hpp"

#include "../../util/rng.hpp"

namespace intra_individual {
    constexpr int max_stack_depth = 128;
    constexpr int reduction_threads_per_block = 32;

    __global__ 
    void weight_update(c_real_1d weight_grads, real_1d weight, int m, float learning_rate) {
        // Shared memory for block-level reduction
        __shared__ float sharedMem[reduction_threads_per_block];
        
        // Thread and block indices
        int tid = threadIdx.x;
        int bid = blockIdx.x;
        int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
        
        // Load data into shared memory
        sharedMem[tid] = (globalIdx < m) ? weight_grads[globalIdx] : 0.0f;
        __syncthreads();
        
        // Parallel reduction within the block
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                sharedMem[tid] += sharedMem[tid + stride];
            }
            __syncthreads();
        }
        
        // First thread of each block writes block sum to output
        if (tid == 0) {
            atomicAdd(weight, -learning_rate * sharedMem[0]);
        }
    }

    __global__
    void vm(c_inst_1d bytecode, 
            const int bytecode_length,
            const int m, 
            c_real_2d X_d, 
            c_real_1d y_d, 
            real_2d stack_d, 
            real_2d intermediate_d,
            c_real_1d weights_d,
            real_2d weights_grad_d,
            const int nweights,
            real_1d loss_d)
    {
        const int tid = blockDim.x * blockIdx.x + threadIdx.x;

        // Reset weight gradients that will be computed by this thread
        for (int j = 0; j < nweights; ++j) {
            weights_grad_d[j][tid] = 0;
        }

        int stack_pointer = 0;
        int intermediate_pointer = 0;

        const StackState s(
            stack_d,
            intermediate_d,
            stack_pointer,
            intermediate_pointer
        );

        int program_counter = 0;

        // Forward propagate and evaluate loss
        vm_debug_print(tid, "Forward propagation");
        vm_control<FORWARD, INTRA_INDIVIDUAL, c_inst_1d, c_real_1d>(
            tid, tid, bytecode, bytecode_length, 
            m, X_d, y_d, 
            s, program_counter, weights_d, weights_grad_d);

        // Print an empty line in between forward propagation output and backpropagation output
        vm_debug_print(tid, "");

        // Save squared difference as the loss
        loss_d[tid] = powf(stack_d[0][tid], 2);

        // Backpropagate
        vm_debug_print(tid, "Backpropagation");
        vm_control<BACK, INTRA_INDIVIDUAL, c_inst_1d, c_real_1d>(
            tid, tid, bytecode, bytecode_length, 
            m, X_d, y_d, 
            s, program_counter, 
            weights_d, weights_grad_d);
    }

    Runner::Runner(std::shared_ptr<Dataset> dataset, const int nweights) : dataset(dataset), nweights(nweights) {
        const int threadsPerBlock = min(dataset->m, hipState.props.maxThreadsPerBlock);
        const int blocks = (dataset->m + threadsPerBlock - 1) / threadsPerBlock;

        gridDim = dim3(blocks);
        blockDim = dim3(threadsPerBlock);

        int reduction_blocks_per_grid = (dataset->m + reduction_threads_per_block - 1) / 
            reduction_threads_per_block;

        reduction_grid_dim = dim3(reduction_blocks_per_grid);
        reduction_block_dim = dim3(reduction_threads_per_block);

        stack_d = std::make_shared<Array2D<float>>(max_stack_depth, dataset->m);
        intermediate_d = std::make_shared<Array2D<float>>(max_stack_depth, dataset->m);
        weights_d = std::make_shared<Array1D<float>>(nweights);
        weights_grad_d = std::make_shared<Array2D<float>>(nweights, dataset->m);
        weights_grad_reduced_sum_d = std::make_shared<Array2D<float>>(nweights, reduction_blocks_per_grid);
        loss_d = std::make_shared<Array1D<float>>(dataset->m);
    }

    void Runner::run(c_inst_1d code, int code_length, int epochs, float learning_rate) {
        for (int i = 0; i < epochs; ++i) {
            // Run forward propagation and backpropagation to compute
            // weight gradients for each weight and for each data point
            hipLaunchKernelGGL(
                vm, gridDim, blockDim, 0, hipState.stream,
                code, code_length, 
                dataset->m, dataset->X_d, dataset->y_d,
                stack_d->ptr, intermediate_d->ptr,
                weights_d->ptr, weights_grad_d->ptr, nweights, loss_d->ptr);

            // For every weight, sum gradient contributions from all data points using reduction
            // Apply gradient descent rule
            for (int i = 0; i < nweights; ++i) {
                hipLaunchKernelGGL(
                    weight_update, reduction_grid_dim, reduction_block_dim, 0, hipState.stream, 
                    weights_grad_d->ptr[i], &weights_d->ptr[i], dataset->m, learning_rate);
            }
        }

        HIP_CALL(hipStreamSynchronize(hipState.stream));
    }

    void Runner::run(std::vector<Expression>& population, int epochs, float learning_rate) {
        // Convert symbolic expressions to bytecode program
        Program program_pop;
        program_create(&program_pop, population);

        // Parallel region to exploit remaining resources with inter-individual parallelism
        for (int i = 0; i < population.size(); ++i)
        {
            auto code = program_pop.bytecode[i];
            auto code_length = program_pop.num_of_instructions[i];

            // If the expression has no weights yet, initialize them randomly
            if (population[i].weights.empty()) {
                for (int j = 0; j < nweights; ++j) {
                    weights_d->ptr[j] = 2 * (thread_local_rng() % RAND_MAX) / (float)RAND_MAX - 1;
                }
            } 
            // If the expression already has weights, use them
            else {
                for (int j = 0; j < nweights; ++j) {
                    weights_d->ptr[j] = population[i].weights[j];
                }
            }

            // Do the work
            run(code, code_length, epochs, learning_rate);

            // Compute total loss
            float total_loss = 0;
            #pragma omp simd reduction(+:total_loss)
            for (int i = 0; i < dataset->m; ++i) {
                total_loss += loss_d->ptr[i];
            }

            // Save loss value to the original expression
            population[i].loss = total_loss;

            // Save weights to the original expression
            population[i].weights.resize(weights_d->dim1);
            for (int j = 0; j < weights_d->dim1; ++j) {
                population[i].weights[j] = weights_d->ptr[j];
            }
        }

        // Destroy bytecode program
        program_destroy(program_pop);
    }
}