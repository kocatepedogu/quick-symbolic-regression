#ifndef ARRAYS_HPP
#define ARRAYS_HPP

#include </usr/lib/clang/20/include/omp.h>
#include <hip/hip_runtime.h>

#include <vector>

// Uncomment to enable GPU memory allocation sanitization
#define SANITIZE_MEMORY

enum MemoryEventType {
    ALLOC,
    FREE
};

struct MemoryEvent {
    MemoryEventType type;
    void *address;
    size_t size;
    size_t line_number;
    const char *file_name;
};

struct MemoryEventLock {
    omp_lock_t lock;

    MemoryEventLock();

    ~MemoryEventLock();

    void setlock();

    void unsetlock();
};

extern MemoryEventLock memory_events_lock;

extern std::vector<MemoryEvent> memory_events;

hipError_t sanitized_alloc(void **ptr, size_t size, size_t line_number, const char *file_name);

hipError_t sanitized_dealloc(void *ptr, size_t line_number, const char *file_name);

template <typename T>
hipError_t sanitized_alloc(T **ptr, size_t size, size_t line_number, const char *file_name) {
    return sanitized_alloc(reinterpret_cast<void **>(ptr), size, line_number, file_name);
}

template <typename T>
hipError_t sanitized_dealloc(T *ptr, size_t line_number, const char *file_name) {
    return sanitized_dealloc(reinterpret_cast<void *>(ptr), line_number, file_name);
}

#ifdef SANITIZE_MEMORY
    #define ALLOC(POINTER, SIZE) sanitized_alloc(POINTER, SIZE, __LINE__, __FILE__)
    #define DEALLOC(POINTER) sanitized_dealloc(POINTER, __LINE__, __FILE__)
#else
    #define ALLOC(POINTER, SIZE) hipMallocManaged(POINTER, SIZE)
    #define DEALLOC(POINTER) hipFree(POINTER)
#endif

#endif // ARRAYS_HPP