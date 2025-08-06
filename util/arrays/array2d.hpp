// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef ARRAY2D_HPP
#define ARRAY2D_HPP

#include "../hip.hpp"

namespace qsr {

template <typename T>
struct Ptr2D {
    int dim1;
    int dim2;

    T *ptr;

    __host__ __device__ T &operator[](int i1, int i2) {
        return ptr[i1 * dim2 + i2];
    }

    __host__ __device__ T *operator[](int i1) {
        return ptr + i1 * dim2;
    }
};

template <typename T>
class Array2DF {
public:
    Ptr2D<T> ptr;

    /// Default constructor
    Array2DF() {
        ptr.dim1 = -1;
        ptr.dim2 = -1;
        ptr.ptr = nullptr;
    }

    /// Constructor
    Array2DF(int dim1, int dim2) {
        ptr.dim1 = dim1;
        ptr.dim2 = dim2;

        HIP_CALL(hipMallocManaged(&ptr.ptr, sizeof *ptr.ptr * dim1 * dim2));
    }

    /// Destructor
    ~Array2DF() {
        HIP_CALL(hipFree(ptr.ptr));
    }

    /// Copy constructor
    Array2DF(const Array2DF &other) {
        HIP_CALL(hipFree(ptr.ptr));

        ptr.dim1 = other.ptr.dim1;
        ptr.dim2 = other.ptr.dim2;

        HIP_CALL(hipMallocManaged(&ptr.ptr, sizeof *ptr.ptr * ptr.dim1 * ptr.dim2));
        HIP_CALL(hipMemcpy(ptr.ptr, other.ptr.ptr, sizeof *ptr.ptr * ptr.dim1 * ptr.dim2, hipMemcpyDefault));
    }

    /// Copy assignment operator
    Array2DF& operator=(const Array2DF& other) {
        if (this != &other) {
            HIP_CALL(hipFree(ptr.ptr));

            ptr.dim1 = other.ptr.dim1;
            ptr.dim2 = other.ptr.dim2;

            HIP_CALL(hipMallocManaged(&ptr.ptr, sizeof *ptr.ptr * ptr.dim1 * ptr.dim2));
            HIP_CALL(hipMemcpy(ptr.ptr, other.ptr.ptr, sizeof *ptr.ptr * ptr.dim1 * ptr.dim2, hipMemcpyDefault));
        }

        return *this;
    }
};



} // namespace qsr

#endif // ARRAY2D_HPP