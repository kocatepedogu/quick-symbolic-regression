// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include <hip/hip_runtime.h>

#include "vm.hpp"
#include "../util.hpp"

__global__
void forward_propagate_kernel(const Instruction* bytecode, int bytecode_length,
                              int m, const float *const *X_d, float *y_predicted_d) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < m; i += stride) {
        for (int program_counter = 0; program_counter < bytecode_length; ++program_counter) {
            Instruction instruction = bytecode[program_counter];

            if (tid == 0)
            switch (instruction.opcode) {
                case PUSH_IMMEDIATE:
                    printf("push %f\n", instruction.value);
                    break;
                case PUSH_VARIABLE:
                    printf("push variable %d\n", instruction.argindex);
                    break;
                case PUSH_PARAMETER:
                    printf("push parameter %d\n", instruction.argindex);
                    break;
                case ADD:
                    printf("add\n");
                    break;
                case SUB:
                    printf("sub\n");
                    break;
                case MUL:
                    printf("mul\n");
                    break;
                case DIV:
                    printf("div\n");
                    break;
                case SIN:
                    printf("sin\n");
                    break;
                case COS:
                    printf("cos\n");
                    break;
                case EXP:
                    printf("exp\n");
                    break;
                case NOP:
                    printf("nop\n");
                    break;
            }
        }
    }
}

void forward_propagate(const Program& program, const Dataset& dataset) {
    float *y_predicted_d;
    HIP_CALL(hipMallocManaged(&y_predicted_d, sizeof *y_predicted_d * dataset.m));

    int deviceId;
    HIP_CALL(hipGetDevice(&deviceId));

    hipDeviceProp_t props;
    HIP_CALL(hipGetDeviceProperties(&props, deviceId));
    int maxThreadsPerBlock = props.maxThreadsPerBlock;

    int threadsPerBlock = maxThreadsPerBlock;
    int blocks = (dataset.m + threadsPerBlock - 1) / threadsPerBlock;

    dim3 gridDim(blocks);
    dim3 blockDim(threadsPerBlock);

    hipLaunchKernelGGL(forward_propagate_kernel, gridDim, blockDim, 0, 0,
                       program.bytecode, program.length, dataset.m, dataset.X_d, y_predicted_d);

    HIP_CALL(hipDeviceSynchronize());
}