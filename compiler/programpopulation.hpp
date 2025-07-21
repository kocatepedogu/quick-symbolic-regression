#ifndef PROGRAMPOPULATION_HPP
#define PROGRAMPOPULATION_HPP

#include "program.hpp"

struct ProgramPopulation {
    /// Programs
    Program **individuals;

    /// Total number of individuals
    int length;

    /** @brief Creates empty program population on GPU memory */
    ProgramPopulation(int length);

    /** @brief Copy assignment operator */
    ProgramPopulation& operator=(const ProgramPopulation& pop);

    /** @brief Copy constructor */
    ProgramPopulation(const ProgramPopulation& pop);

    /** @brief Deletes program population from GPU memory */
    ~ProgramPopulation();
};

#endif