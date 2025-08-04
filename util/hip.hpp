// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef UTIL_HPP
#define UTIL_HPP

#include <hip/hip_runtime.h>
#include <iostream>

#define HIP_CALL(call) \
    do { \
        hipError_t error = call; \
        if (error != hipSuccess) { \
            std::cerr << "HIP Error: " << std::endl \
                      << "  File:     " << __FILE__ << std::endl \
                      << "  Line:     " << __LINE__ << std::endl \
                      << "  Function: " << __func__ << std::endl \
                      << "  Error:    " << hipGetErrorString(error) << std::endl \
                      << "  Code:     " << error << std::endl; \
            std::cerr << "HIP API call failed: " << std::string(hipGetErrorString(error)) << std::endl; \
            abort(); \
        } \
    } while(0)

namespace qsr {

template <typename T>
static inline void init_arr_1d(T *&ptr, int dim1) {
    HIP_CALL(hipMallocManaged(&ptr, sizeof(T) * dim1));
}

template <typename T>
static inline void del_arr_1d(T *ptr) {
    HIP_CALL(hipFree(ptr));
}

template <typename T>
static inline void init_arr_2d(T **&ptr, int dim1, int dim2) {
    HIP_CALL(hipMallocManaged(&ptr, sizeof *ptr * dim1));
    for (int i = 0; i < dim1; ++i) {
        HIP_CALL(hipMallocManaged(&ptr[i], sizeof **ptr * dim2));
    }
}

template <typename T>
static inline void del_arr_2d(T **ptr, int dim1) {
    for (int i = 0; i < dim1; ++i) {
        HIP_CALL(hipFree(ptr[i]));
    }
    HIP_CALL(hipFree(ptr));
}

template <typename T>
struct Array1D {
    T *ptr;

    Array1D(int dim1) : dim1(dim1) {
        init_arr_1d(ptr, dim1);
    }

    ~Array1D() {
        HIP_CALL(hipFree(ptr));
    }

    int dim1;
};

template <typename T>
struct Array2D {
    T **ptr;
    
    Array2D(int dim1, int dim2) : dim1(dim1), dim2(dim2) {
        init_arr_2d(ptr, dim1, dim2);
    }

    ~Array2D() {
        del_arr_2d(ptr, dim1);
    }

    int dim1;
    int dim2;
};

struct HIPState {
    int device_id;
    hipDeviceProp_t props;
    hipStream_t stream;

    constexpr HIPState() {
        // Create stream
        HIP_CALL(hipStreamCreate(&stream));

        // Get device properties
        HIP_CALL(hipGetDevice(&device_id));
        HIP_CALL(hipGetDeviceProperties(&props, device_id));
    }

    constexpr ~HIPState() {
        HIP_CALL(hipStreamDestroy(stream));
    }
};

}

#endif