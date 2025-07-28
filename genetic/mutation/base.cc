#include "base.hpp"
#include <cstdio>

Expression BaseMutation::mutate(const Expression &expr) noexcept {
    fprintf(stderr, "Unimplemented base method called.");
    abort();
}