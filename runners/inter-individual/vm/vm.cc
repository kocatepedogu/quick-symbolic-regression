#include <hip/hip_runtime.h>
#include "vm.hpp"

#include "../../../vm/vm_debug.hpp"
#include "../../../vm/vm_types.hpp"
#include "../../../vm/vm_control.hpp"

#include "../../../util/hip.hpp"

namespace inter_individual {
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
                    vm_control<FORWARD, INTER_INDIVIDUAL, c_inst_2d, c_real_2d>(tid, datapoint_idx, bytecode, max_num_of_instructions, m, X_d, y_d, s, program_counter, weights_d, weights_grad_d);

                    // Print an empty line in between forward propagation output and backpropagation output
                    vm_debug_print(tid, "");

                    // Add squared difference to total loss
                    total_loss += powf(stack_d[0][tid], 2);

                    // Backpropagate
                    vm_debug_print(tid, "Backpropagation");
                    vm_control<BACK, INTER_INDIVIDUAL, c_inst_2d, c_real_2d>(tid, datapoint_idx, bytecode, max_num_of_instructions, m, X_d, y_d, s, program_counter, weights_d, weights_grad_d);
                }

                // Apply weight updates (gradient descent)
                for (int weight_idx = 0; weight_idx < nweights; ++weight_idx) {
                    weights_d[weight_idx][tid] -= learning_rate * weights_grad_d[weight_idx][tid];
                }
            }

            loss_d[tid] = total_loss;
        }
    }

    VirtualMachine::VirtualMachine(const Dataset& dataset, int nweights, hipStream_t &stream) :
        dataset(dataset), nweights(nweights), stream(stream)
    {
        HIP_CALL(hipGetDevice(&device_id));
        HIP_CALL(hipGetDeviceProperties(&props, device_id));
    }

    void VirtualMachine::fit(const Program &program, real_1d loss_d, int epochs, float learning_rate) {
        real_2d_mut stack_d;
        real_2d_mut intermediate_d;

        real_2d_mut weights_d;
        real_2d_mut weights_grad_d;

        /* 
        * Decide number of blocks and threads for computation
        * - The total number of threads must be greater than or equal to 
        *   the number of individuals
        * - Each block will have the maximum number of threads supported by
        *   the device (probably 1024).
        * - Excess threads are later masked inside kernel with if (tid < program.num_of_individuals).
        */

        const int threadsPerBlock = min(program.num_of_individuals, props.maxThreadsPerBlock);
        const int blocks = (program.num_of_individuals + threadsPerBlock - 1) / threadsPerBlock;

        dim3 gridDim(blocks);
        dim3 blockDim(threadsPerBlock);

        /* 
        * Allocate array stack_d as stack memory for bytecode virtual machine.
        * - Each thread accesses its own stack. 
        *   The dimensions of the stack are [max_stack_depth][num_threads]
        */
        init_arr_2d(stack_d, max_stack_depth, program.num_of_individuals);

        /*
        * Allocate array intermediate_d to store intermediate calculation 
        * results for later use in backpropagation. The dimensions of 
        * intermediate_d is the same as stack_d.
        */
        init_arr_2d(intermediate_d, max_stack_depth, program.num_of_individuals);

        /*
        * Allocate array weights_d.
        * - Each thread fits a different expression and has a different set of weight values.
        * - The array has dimensions [nweights][num_threads].
        */
        init_arr_2d(weights_d, nweights, program.num_of_individuals);
        for (int i = 0; i < nweights; ++i) {
            for (int j = 0; j < program.num_of_individuals; ++j) {
                weights_d[i][j] = 0.5;
            }
        }

        /*
        * Allocate array weights_grad_d.
        * - The array has the same dimensions as weights_d: [nweights][num_threads]. 
        * - i'th weight gradient of the j'th expression is (sequentially) accumulated in weights_grad_d[i][j]
        */
        init_arr_2d(weights_grad_d, nweights, program.num_of_individuals);

        hipLaunchKernelGGL(
            vm, gridDim, blockDim, 0, stream,
            program.bytecode, 
            program.max_num_of_instructions, 
            program.num_of_individuals,
            dataset.m, dataset.X_d, dataset.y_d,
            stack_d, intermediate_d,
            weights_d, weights_grad_d, nweights,
            epochs, learning_rate, loss_d);

        HIP_CALL(hipStreamSynchronize(stream));

        // Deallocate array intermediate_d.
        del_arr_2d(intermediate_d, max_stack_depth);

        // Deallocate array stack_d
        del_arr_2d(stack_d, max_stack_depth);

        // Deallocate array weights_d
        del_arr_2d(weights_d, nweights);

        // Deallocate array weights_grad_d
        del_arr_2d(weights_grad_d, nweights);
    }
};