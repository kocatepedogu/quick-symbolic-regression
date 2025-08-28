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