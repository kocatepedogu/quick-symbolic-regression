#include "common/testdata.hpp"

#include "../genetic/genetic_programming.hpp"

#include <cmath>

int main(void) {
    float **X, *y;

    // Generate ground truth data
    generate_test_data(X, y, [](float x) { 
        return 2.5382 * cos(x)*x + x*x - 0.5; 
    });

    // Create dataset
    Dataset dataset(X, y, test_data_length, 1);

    #pragma omp parallel num_threads(10)
    {
        // Create island
        GeneticProgramming island(dataset, 2, 2000);

        // Iterate island
        island.iterate(10);
    }

    return 0;
}