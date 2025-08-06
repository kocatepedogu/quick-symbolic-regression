// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef ARRAY1D_HPP
#define ARRAY1D_HPP

#include "../hip.hpp"

namespace qsr {
    template <typename T>
    struct Array1D {
        T *ptr;
        int dim1;

        Array1D() {
            ptr = nullptr;
            dim1 = -1;
        }

        Array1D(int dim1) : dim1(dim1) {
            HIP_CALL(hipMallocManaged(&ptr, sizeof(T) * dim1));
        }

        ~Array1D() {
            HIP_CALL(hipFree(ptr));
        }

        // Copy constructor
        Array1D(const Array1D &other) {
            HIP_CALL(hipFree(ptr));

            this->dim1 = other.dim1;

            HIP_CALL(hipMallocManaged(&ptr, sizeof(T) * dim1));
            HIP_CALL(hipMemcpy(ptr, other.ptr, sizeof(T) * dim1, hipMemcpyDefault));
        }

        // Copy assignment operator
        Array1D& operator=(const Array1D& other) {
            if (this != &other) {
                HIP_CALL(hipFree(ptr));

                this->dim1 = other.dim1;

                HIP_CALL(hipMallocManaged(&ptr, sizeof(T) * dim1));
                HIP_CALL(hipMemcpy(ptr, other.ptr, sizeof(T) * dim1, hipMemcpyDefault));    
            }

            return *this;
        }
    };
};

#endif // ARRAY1D_HPP