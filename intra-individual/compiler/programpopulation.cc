#include "programpopulation.hpp"

#include "../../compiler/compiler.hpp"

namespace intra_individual {
    ProgramPopulation::ProgramPopulation(int length) : length(length) {
        individuals = new Program*[length];
    }

    ProgramPopulation& ProgramPopulation::operator=(const ProgramPopulation& pop) {
        // Delete target memory
        for (int i = 0; i < this->length; ++i) {
            delete this->individuals[i];
        }
        delete[] individuals;

        // Update length
        this->length = pop.length;

        // Allocate new target memory and copy from source
        this->individuals = new Program*[pop.length];
        for (int i = 0; i < pop.length; ++i) {
            this->individuals[i] = new Program(*pop.individuals[i]);
        }

        // Return self
        return *this;
    }

    ProgramPopulation::ProgramPopulation(const ProgramPopulation& pop) {
        // Delete target memory
        for (int i = 0; i < this->length; ++i) {
            delete this->individuals[i];
        }
        delete[] individuals;

        // Update length
        this->length = pop.length;

        // Allocate new target memory and copy from source
        this->individuals = new Program*[pop.length];
        for (int i = 0; i < pop.length; ++i) {
            this->individuals[i] = new Program(*pop.individuals[i]);
        }
    }

    ProgramPopulation::~ProgramPopulation() {
        for (int i = 0; i < this->length; ++i) {
            delete this->individuals[i];
        }
        delete[] individuals;
    }

    ProgramPopulation compile(const std::vector<Expression>& exp_pop) noexcept {
        ProgramPopulation programPopulation(exp_pop.size());
        for (int i = 0; i < exp_pop.size(); ++i) {
            programPopulation.individuals[i] = new Program(compile(exp_pop[i]));
        }
        return programPopulation;
    }
}