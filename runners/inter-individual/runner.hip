#include "runner.hpp"

#include <hip/hip_runtime.h>

#include "../../../vm/vm_debug.hpp"
#include "../../../vm/vm_types.hpp"
#include "../../../vm/vm_control.hpp"
#include "../../../util/rng.hpp"
#include "../../../util/hip.hpp"

#include "./program/program.hpp"

#include <hip/hip_runtime.h>

namespace inter_individual {
    Runner::Runner(std::shared_ptr<Dataset> dataset, int nweights) :
        dataset(dataset), nweights(nweights) {}

    constexpr int max_stack_depth = 128;

    __global__
    void vm(c_inst_2d bytecode, 
            const int max_num_of_instructions,
            const int num_of_individuals,
            const int m, 
            c_real_2d X_d, 
            c_real_1d y_d, 
            real_2d stack_d, 
            real_2d intermediate_d,
            real_2d weights_d,
            real_2d weights_grad_d,
            const int nweights,
            const int nepochs,
            const float learning_rate,
            real_1d loss_d)
    {
        const int tid = blockDim.x * blockIdx.x + threadIdx.x;

        if (tid < num_of_individuals) {
            float total_loss = 0;

            for (int epoch = 0; epoch < nepochs; ++epoch) {
                // Reset weight gradients
                for (int weight_idx = 0; weight_idx < nweights; ++weight_idx) {
                    weights_grad_d[weight_idx][tid] = 0;
                }

                total_loss = 0;

                // For all datapoints, compute and sum weight gradients
                for (int datapoint_idx = 0; datapoint_idx < m; ++datapoint_idx) {
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
                    vm_control<FORWARD, INTER_INDIVIDUAL, c_inst_2d, c_real_2d>(
                        tid, datapoint_idx, bytecode, max_num_of_instructions, 
                        m, X_d, y_d, 
                        s, program_counter, 
                        weights_d, weights_grad_d);

                    // Print an empty line in between forward propagation output and backpropagation output
                    vm_debug_print(tid, "");

                    // Add squared difference to total loss
                    total_loss += powf(stack_d[0][tid], 2);

                    // Backpropagate
                    vm_debug_print(tid, "Backpropagation");
                    vm_control<BACK, INTER_INDIVIDUAL, c_inst_2d, c_real_2d>(
                        tid, datapoint_idx, bytecode, max_num_of_instructions, 
                        m, X_d, y_d, 
                        s, program_counter, 
                        weights_d, weights_grad_d);
                }

                // Apply weight updates (gradient descent)
                for (int weight_idx = 0; weight_idx < nweights; ++weight_idx) {
                    weights_d[weight_idx][tid] -= learning_rate * weights_grad_d[weight_idx][tid];
                }
            }

            loss_d[tid] = total_loss;
        }
    }

    VirtualMachineResult Runner::run(const Program &program, int epochs, float learning_rate) {
        const int threadsPerBlock = min(program.num_of_individuals, config.props.maxThreadsPerBlock);
        const int blocks = (program.num_of_individuals + threadsPerBlock - 1) / threadsPerBlock;

        dim3 gridDim(blocks);
        dim3 blockDim(threadsPerBlock);

        auto loss_d = std::make_unique<Array1D<float>>(program.num_of_individuals);
        auto stack_d = std::make_unique<Array2D<float>>(max_stack_depth, program.num_of_individuals);
        auto intermediate_d = std::make_unique<Array2D<float>>(max_stack_depth, program.num_of_individuals);
        auto weights_d = std::make_unique<Array2D<float>>(nweights, program.num_of_individuals);
        auto weights_grad_d = std::make_unique<Array2D<float>>(nweights, program.num_of_individuals);

        for (int i = 0; i < nweights; ++i) {
            for (int j = 0; j < program.num_of_individuals; ++j) {
                weights_d->ptr[i][j] = 2 * (thread_local_rng() % RAND_MAX) / (float)RAND_MAX - 1;
            }
        }

        hipLaunchKernelGGL(
            vm, gridDim, blockDim, 0, config.stream,
            program.bytecode, 
            program.max_num_of_instructions, 
            program.num_of_individuals,
            dataset->m, dataset->X_d, dataset->y_d,
            stack_d->ptr, intermediate_d->ptr,
            weights_d->ptr, weights_grad_d->ptr, nweights,
            epochs, learning_rate, loss_d->ptr);

        HIP_CALL(hipStreamSynchronize(config.stream));

        // Return learned weights and loss for each expression
        return VirtualMachineResult {
            std::move(weights_d),
            std::move(loss_d)
        };
    }

    void Runner::run(std::vector<Expression>& population, int epochs, float learning_rate) {
        // Convert symbolic expression to bytecode program
        Program program_pop;
        program_create(&program_pop, population);

        // Run virtual machine
        auto result = run(program_pop, epochs, learning_rate);

        // Destroy programs
        program_destroy(program_pop);

        // Write loss values to the original expressions
        for (int i = 0; i < population.size(); ++i) {
            population[i].loss = result.loss_d->ptr[i];
        }

        // Write weight values to the original expressions
        for (int i = 0; i < population.size(); ++i) {
            population[i].weights.resize(nweights);
            for (int j = 0; j < nweights; ++j) {
                population[i].weights[j] = result.weights_d->ptr[j][i];
            }
        }
    }
}
