#include "util/arrays/memory.hpp"
#include <iostream>

MemoryEventLock memory_events_lock;

std::vector<MemoryEvent> memory_events;

MemoryEventLock::MemoryEventLock() {
    omp_init_lock(&lock);
}

MemoryEventLock::~MemoryEventLock() {
    omp_destroy_lock(&lock);
}

void MemoryEventLock::setlock() {
    omp_set_lock(&lock);
}

void MemoryEventLock::unsetlock() {
    omp_unset_lock(&lock);
}

hipError_t sanitized_alloc(void **ptr, size_t size, size_t line_number, const char *file_name) {
    // Lock the memory event list
    memory_events_lock.setlock();

    // Allocate memory
    hipError_t result = hipMallocManaged(ptr, size);

    // Record the allocation
    memory_events.push_back({ALLOC, *ptr, size, line_number, file_name});

    // Unlock the memory event list
    memory_events_lock.unsetlock();

    // Return the result of the HIP call
    return result;
}

hipError_t sanitized_dealloc(void *ptr, size_t line_number, const char *file_name) {
    // Lock the memory event list
    memory_events_lock.setlock();

    // Consider deallocating nullptr an error (although HIP allows it)
    if (ptr == nullptr) {
        std::cerr << "Error: Try to free null pointer at " 
                  << file_name << ":" << line_number << std::endl;
        abort();
    }

    // Check if the same address was freed without an allocation afterwards
    for (auto it = memory_events.rbegin(); it != memory_events.rend(); ++it) {
        if (it->type == ALLOC && it->address == ptr) {
            // Found a matching allocation
            break;
        }

        if (it->type == FREE && it->address == ptr) {
            // Found a previous free which is not followed by an allocation
            std::cerr << "Error: Address " << ptr << " was already freed at " 
                      << it->file_name << ":" << it->line_number 
                      << " without an allocation afterwards." << std::endl;
            abort();
        }
    }

    // Free the memory
    hipError_t result = hipFree(ptr);

    // Record the deallocation
    memory_events.push_back({FREE, ptr, 0, line_number, file_name});

    // Unlock the memory event list
    memory_events_lock.unsetlock();

    // Return the result of the HIP call
    return result;
}
