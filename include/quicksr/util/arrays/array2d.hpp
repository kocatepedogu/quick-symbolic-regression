// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef ARRAY2D_HPP
#define ARRAY2D_HPP

#include "util/hip.hpp"
#include "array1d.hpp"
#include "util/arrays/memory.hpp"

namespace qsr {

template <typename T>
struct Ptr2D {
    int dim1;
    int dim2;

    T *__restrict__ ptr;

    __host__ __device__ T &operator[](int i1, int i2) const {
        #ifdef SANITIZE_MEMORY
        if (dim1 <= 0) {
            printf("Error [%s:%d]: Array2D dimension 1 is not positive (dim1=%d)\n", __FILE__, __LINE__, dim1);
            abort();
        }

        if (dim2 <= 0) {
            printf("Error [%s:%d]: Array2D dimension 2 is not positive (dim2=%d)\n", __FILE__, __LINE__, dim2);
            abort();
        }

        if (ptr == nullptr) {
            printf("Error [%s:%d]: Array2D pointer is not initialized\n", __FILE__, __LINE__);
            abort();
        }

        if (i1 >= dim1) {
            printf("Error [%s:%d]: Array2D index 1 out of bounds (%d >= %d)\n", __FILE__, __LINE__, i1, dim1);
            abort();
        }

        if (i2 >= dim2) {
            printf("Error [%s:%d]: Array2D index 2 out of bounds (%d >= %d)\n", __FILE__, __LINE__, i2, dim2);
            abort();
        }
        #endif

        return ptr[i1 * dim2 + i2];
    }

    __host__ __device__ T *__restrict__ operator[](int i1) const {
        #ifdef SANITIZE_MEMORY
        if (dim1 <= 0) {
            printf("Error [%s:%d]: Array2D dimension 1 is not positive (dim1=%d)\n", __FILE__, __LINE__, dim1);
            abort();
        }

        if (dim2 <= 0) {
            printf("Error [%s:%d]: Array2D dimension 2 is not positive (dim2=%d)\n", __FILE__, __LINE__, dim2);
            abort();
        }

        if (ptr == nullptr) {
            printf("Error [%s:%d]: Array2D pointer is not initialized\n", __FILE__, __LINE__);
            abort();
        }

        if (i1 >= dim1) {
            printf("Error [%s:%d]: Array2D index 1 out of bounds (%d >= %d)\n", __FILE__, __LINE__, i1, dim1);
            abort();
        }
        #endif

        return ptr + i1 * dim2;
    }

    __host__ __device__ Ptr1D<T> operator()(int i1) const {
        return Ptr1D<T>{dim2, ptr + i1 * dim2};
    }
};

template <typename T>
class Array2D {
public:
    Ptr2D<T> ptr;

    /// Default constructor
    Array2D() {
        ptr.dim1 = 0;
        ptr.dim2 = 0;
        ptr.ptr = nullptr;
    }

    /// Constructor
    Array2D(int dim1, int dim2) {
        ptr.dim1 = dim1;
        ptr.dim2 = dim2;
        ptr.ptr = nullptr;

        if (dim1 > 0 && dim2 > 0) {
            void *newptr;
            HIP_CALL(ALLOC(&newptr, sizeof *ptr.ptr * dim1 * dim2));
            ptr.ptr = (T *)(newptr);
        }
    }

    /// Destructor
    ~Array2D() {
        if (ptr.ptr != nullptr) {
            HIP_CALL(DEALLOC(ptr.ptr));
        }

        ptr.dim1 = 0;
        ptr.dim2 = 0;
        ptr.ptr = nullptr;
    }

    /// Copy constructor
    Array2D(const Array2D &other) {
        if (ptr.ptr != nullptr) {
            HIP_CALL(DEALLOC(ptr.ptr));
        }

        ptr.dim1 = other.ptr.dim1;
        ptr.dim2 = other.ptr.dim2;
        ptr.ptr = nullptr;

        if (ptr.dim1 > 0 && ptr.dim2 > 0) {
            void *newptr;
            HIP_CALL(ALLOC(&newptr, sizeof *ptr.ptr * ptr.dim1 * ptr.dim2));
            ptr.ptr = (T *)(newptr);

            HIP_CALL(hipMemcpy(ptr.ptr, other.ptr.ptr, sizeof *ptr.ptr * ptr.dim1 * ptr.dim2, hipMemcpyDefault));
        }
    }

    /// Copy assignment operator
    Array2D& operator=(const Array2D& other) {
        if (this != &other) {
            if (ptr.ptr != nullptr) {
                HIP_CALL(DEALLOC(ptr.ptr));
            }

            ptr.dim1 = other.ptr.dim1;
            ptr.dim2 = other.ptr.dim2;
            ptr.ptr = nullptr;

            if (ptr.dim1 > 0 && ptr.dim2 > 0) {
                void *newptr;
                HIP_CALL(ALLOC(&newptr, sizeof *ptr.ptr * ptr.dim1 * ptr.dim2));
                ptr.ptr = (T *)(newptr);

                HIP_CALL(hipMemcpy(ptr.ptr, other.ptr.ptr, sizeof *ptr.ptr * ptr.dim1 * ptr.dim2, hipMemcpyDefault));
            }
        }

        return *this;
    }

    /// Resizes array. Contents are discarded.
    void resize(int new_dim1, int new_dim2) {
        if (new_dim1 > ptr.dim1 || new_dim2 > ptr.dim2) {
            if (ptr.ptr != nullptr) {
                HIP_CALL(DEALLOC(ptr.ptr));
            }

            // Allocate the larger array
            ptr.dim1 = new_dim1;
            ptr.dim2 = new_dim2;
            ptr.ptr = nullptr;

            if (ptr.dim1 > 0 && ptr.dim2 > 0) {
                void *newptr;
                HIP_CALL(ALLOC(&newptr, sizeof *ptr.ptr * ptr.dim1 * ptr.dim2));
                ptr.ptr = (T *)(newptr);
            }
        }
    }
};


} // namespace qsr

#endif // ARRAY2D_HPP