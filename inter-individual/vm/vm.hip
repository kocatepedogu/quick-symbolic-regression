#include <hip/hip_runtime.h>
#include "vm.hpp"

#include "../../vm/vm_debug.hpp"
#include "../../vm/vm_types.hpp"

#include "../../util.hpp"

namespace inter_individual {
    constexpr int max_stack_depth = 32;

    __global__
    void vm(Instruction **bytecode, 
            const int max_num_of_instructions,
            const int num_of_individuals,
            const int m, 
            const float *const __restrict__ *const __restrict__ X_d, 
            const float *const __restrict__ y_d, 
            float *const __restrict__ *const __restrict__ stack_d, 
            float *const __restrict__ *const __restrict__ intermediate_d) 
    {
        const int tid = blockDim.x * blockIdx.x + threadIdx.x;

        vm_debug_print(tid, "test\n");
    }

    VirtualMachine::VirtualMachine(const Dataset& dataset) :
        dataset(dataset)
    {
        HIP_CALL(hipGetDevice(&device_id));
        HIP_CALL(hipGetDeviceProperties(&props, device_id));
    }

    void VirtualMachine::fit(const Program &program) {
        float **stack_d;
        float **intermediate_d;

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

        hipLaunchKernelGGL(
            vm, gridDim, blockDim, 0, 0,
            program.bytecode, 
            program.max_num_of_instructions, 
            program.num_of_individuals,
            dataset.m, dataset.X_d, dataset.y_d,
            stack_d, intermediate_d);

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
    }
};