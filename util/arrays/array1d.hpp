// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef ARRAY1D_HPP
#define ARRAY1D_HPP

#include "../hip.hpp"

namespace qsr {
    template <typename T>
    struct Ptr1D {
        int dim1;
        T *ptr;

        __host__ __device__ T &operator[](int i) {
            return ptr[i];
        }
    };

    template <typename T>
    struct Array1D {
        Ptr1D<T> ptr;

        Array1D() {
            ptr.ptr = nullptr;
            ptr.dim1 = -1;
        }

        Array1D(int dim1) {
            ptr.dim1 = dim1;
            HIP_CALL(hipMallocManaged(&ptr.ptr, sizeof(T) * ptr.dim1));
        }

        ~Array1D() {
            HIP_CALL(hipFree(ptr.ptr));
        }

        // Copy constructor
        Array1D(const Array1D &other) {
            HIP_CALL(hipFree(ptr.ptr));

            this->ptr.dim1 = other.ptr.dim1;

            HIP_CALL(hipMallocManaged(&ptr.ptr, sizeof(T) * ptr.dim1));
            HIP_CALL(hipMemcpy(ptr.ptr, other.ptr.ptr, sizeof(T) * ptr.dim1, hipMemcpyDefault));
        }

        // Copy assignment operator
        Array1D& operator=(const Array1D& other) {
            if (this != &other) {
                HIP_CALL(hipFree(ptr.ptr));

                this->ptr.dim1 = other.ptr.dim1;

                HIP_CALL(hipMallocManaged(&ptr.ptr, sizeof(T) * ptr.dim1));
                HIP_CALL(hipMemcpy(ptr.ptr, other.ptr.ptr, sizeof(T) * ptr.dim1, hipMemcpyDefault));    
            }

            return *this;
        }
    };
};

#endif // ARRAY1D_HPP