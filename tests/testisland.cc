#include "common/testdata.hpp"

#include "../genetic/genetic_programming.hpp"

#include "../genetic/initializer/default.hpp"
#include "../genetic/mutation/default.hpp"
#include "../genetic/crossover/default.hpp"
#include "../genetic/selection/fitness_proportional_selection.hpp"

#include <cmath>

int main(void) {
    float **X, *y;

    // Generate ground truth data
    generate_test_data(X, y, [](float x) { 
        return 2.5382 * cos(x)*x + x*x - 0.5; 
    });

    // Create dataset
    Dataset dataset(X, y, test_data_length, 1);

    // Genetic operators
    DefaultInitializer initializer(1, 2, 3, 2000);
    DefaultMutation mutation(1, 2, 3, 0.7);
    DefaultCrossover crossover(1.0);
    FitnessProportionalSelection selection;

    #pragma omp parallel num_threads(10)
    {
        // Create island
        GeneticProgramming genetic_programming(
            dataset, 
            2, 
            2000, 
            initializer, 
            mutation, 
            crossover, 
            selection);

        // Iterate island
        genetic_programming.iterate(10);
    }

    return 0;
}