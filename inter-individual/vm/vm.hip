#include <hip/hip_runtime.h>
#include "vm.hpp"

#include "../../vm/vm_debug.hpp"
#include "../../vm/vm_types.hpp"
#include "../../vm/vm_control.hpp"

#include "../../util.hpp"

namespace inter_individual {
    constexpr int max_stack_depth = 32;

    __global__
    void vm(c_inst_2d bytecode, 
            const int max_num_of_instructions,
            const int num_of_individuals,
            const int m, 
            c_real_2d X_d, 
            c_real_1d y_d, 
            real_2d stack_d, 
            real_2d intermediate_d,
            c_real_2d weights_d,
            real_2d weights_grad_d,
            const int nweights)
    {
        const int tid = blockDim.x * blockIdx.x + threadIdx.x;

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
        vm_control<FORWARD, INTER_INDIVIDUAL, c_inst_2d, c_real_2d>(tid, 0, bytecode, max_num_of_instructions, m, X_d, y_d, s, program_counter, weights_d, weights_grad_d);
    }

    VirtualMachine::VirtualMachine(const Dataset& dataset, int nweights) :
        dataset(dataset), nweights(nweights)
    {
        HIP_CALL(hipGetDevice(&device_id));
        HIP_CALL(hipGetDeviceProperties(&props, device_id));
    }

    void VirtualMachine::fit(const Program &program) {
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
        HIP_CALL(hipMallocManaged(&stack_d, sizeof *stack_d * max_stack_depth));
        for (int i = 0; i < max_stack_depth; ++i) {
            HIP_CALL(hipMallocManaged(&stack_d[i], sizeof **stack_d * program.num_of_individuals));
        }

        /*
        * Allocate array intermediate_d to store intermediate calculation 
        * results for later use in backpropagation. The dimensions of 
        * intermediate_d is the same as stack_d.
        */
        HIP_CALL(hipMallocManaged(&intermediate_d, sizeof *intermediate_d * max_stack_depth));
        for (int i = 0; i < max_stack_depth; ++i) {
            HIP_CALL(hipMallocManaged(&intermediate_d[i], sizeof **intermediate_d * program.num_of_individuals));
        }

        /*
        * Allocate array weights_d.
        * - Each thread fits a different expression and has a different set of weight values.
        * - The array has dimensions [nweights][num_threads].
        */
        HIP_CALL(hipMallocManaged(&weights_d, sizeof *weights_d * nweights));
        for (int i = 0; i < nweights; ++i) {
            HIP_CALL(hipMallocManaged(&weights_d[i], sizeof **weights_d * program.num_of_individuals));
        }

        /*
        * Allocate array weights_grad_d.
        * - The array has the same dimensions as weights_d. 
        * - i'th weight gradient of the j'th expression is (sequentially) accumulated in weights_grad_d[i][j]
        */
        HIP_CALL(hipMallocManaged(&weights_grad_d, sizeof *weights_grad_d * nweights));
        for (int i = 0; i < nweights; ++i) {
            HIP_CALL(hipMallocManaged(&weights_grad_d[i], sizeof **weights_grad_d * dataset.m));
        }

        hipLaunchKernelGGL(
            vm, gridDim, blockDim, 0, 0,
            program.bytecode, 
            program.max_num_of_instructions, 
            program.num_of_individuals,
            dataset.m, dataset.X_d, dataset.y_d,
            stack_d, intermediate_d,
            weights_d, weights_grad_d, nweights);

        HIP_CALL(hipDeviceSynchronize());

        // Deallocate array intermediate_d.
        for (int i = 0; i < max_stack_depth; ++i) {
            HIP_CALL(hipFree(intermediate_d[i]));
        }
        HIP_CALL(hipFree(intermediate_d));

        // Deallocate array stack_d
        for (int i = 0; i < max_stack_depth; ++i) {
            HIP_CALL(hipFree(stack_d[i]));
        }
        HIP_CALL(hipFree(stack_d));

        // Deallocate array weights_d
        for (int i = 0; i < nweights; ++i) {
            HIP_CALL(hipFree(weights_d[i]));
        }
        HIP_CALL(hipFree(weights_d));

        // Deallocate array weights_grad_d
        for (int i = 0; i < nweights; ++i) {
            HIP_CALL(hipFree(weights_grad_d[i]));
        }
        HIP_CALL(hipFree(weights_grad_d));
    }
};