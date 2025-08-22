// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef ARRAY1D_HPP
#define ARRAY1D_HPP

#include "../hip.hpp"
#include "memory.hpp"

namespace qsr {
    template <typename T>
    struct Ptr1D {
        int dim1;
        T *ptr;

        __host__ __device__ T &operator[](int i) const {
            #ifdef SANITIZE_MEMORY
            if (dim1 <= 0) {
                printf("Error [%s:%d]: Array1D dimension is not positive (dim1=%d)\n", __FILE__, __LINE__, dim1);
                abort();
            }

            if (ptr == nullptr) {
                printf("Error [%s:%d]: Array1D pointer is not initialized\n", __FILE__, __LINE__);
                abort();
            }

            if (i >= dim1) {
                printf("Error [%s:%d]: Array1D index out of bounds (%d >= %d)\n", __FILE__, __LINE__, i, dim1);
                abort();
            }
            #endif

            return ptr[i];
        }
    };

    template <typename T>
    struct Array1D {
        Ptr1D<T> ptr;

        Array1D() {
            ptr.ptr = nullptr;
            ptr.dim1 = 0;
        }

        Array1D(int dim1) {
            ptr.dim1 = dim1;
            HIP_CALL(ALLOC(&ptr.ptr, sizeof(T) * ptr.dim1));
        }

        ~Array1D() {
            if (ptr.ptr != nullptr) {
                HIP_CALL(DEALLOC(ptr.ptr));
            }
        }

        // Copy constructor
        Array1D(const Array1D &other) {
            if (ptr.ptr != nullptr) {
                HIP_CALL(DEALLOC(ptr.ptr));
            }

            this->ptr.dim1 = other.ptr.dim1;

            HIP_CALL(ALLOC(&ptr.ptr, sizeof(T) * ptr.dim1));
            HIP_CALL(hipMemcpy(ptr.ptr, other.ptr.ptr, sizeof(T) * ptr.dim1, hipMemcpyDefault));
        }

        // Copy assignment operator
        Array1D& operator=(const Array1D& other) {
            if (this != &other) {
                if (ptr.ptr != nullptr) {
                    HIP_CALL(DEALLOC(ptr.ptr));
                }

                this->ptr.dim1 = other.ptr.dim1;

                HIP_CALL(ALLOC(&ptr.ptr, sizeof(T) * ptr.dim1));
                HIP_CALL(hipMemcpy(ptr.ptr, other.ptr.ptr, sizeof(T) * ptr.dim1, hipMemcpyDefault));    
            }

            return *this;
        }

        // Resizes the array. Contents are discarded.
        void resize(int new_dim1) {
            if (new_dim1 > ptr.dim1) {
                if (ptr.ptr != nullptr) {
                    HIP_CALL(DEALLOC(ptr.ptr));
                }

                // Allocate the larger array                
                ptr.dim1 = new_dim1;
                ptr.ptr = nullptr;

                HIP_CALL(ALLOC(&ptr.ptr, sizeof(T) * ptr.dim1));
            }
        }
    };
};

#endif // ARRAY1D_HPP