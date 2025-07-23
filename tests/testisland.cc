#include "common/testdata.hpp"

#include "../genetic/island.hpp"

#include <cmath>

int main(void) {
    float **X, *y;

    // Generate ground truth data
    generate_test_data(X, y, [](float x) { 
        return 2.5382 * cos(x)*x + x*x - 0.5; 
    });

    // Create dataset
    Dataset dataset(X, y, test_data_length, 1);

    // Create island
    Island island(dataset, 2, 1000);

    // Iterate island
    island.iterate(20);

    return 0;
}