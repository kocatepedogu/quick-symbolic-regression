#include "expression_comparator.hpp"

#include <cmath>

bool ExpressionComparator::operator() (const Expression& a, const Expression& b) {
    // If a is equal to b, a < b is false.
    if (a == b) {
        return false;
    }

    // If a is NaN, but b is not NaN, treat a as the worst solution.
    if (std::isnan(a.loss) && !std::isnan(b.loss)) {
        return true;
    }

    // If a is not NaN, but b is NaN, treat b as the worst solution
    if (!std::isnan(a.loss) && std::isnan(b.loss)) {
        return false;
    }

    // If both are NaN, NaN < NaN is false
    if (std::isnan(a.loss) && std::isnan(b.loss)) {
        return false;
    }

    // If a is more complex
    if (a.num_of_nodes > b.num_of_nodes) {
        // If a is more complex, but b has significantly higher loss
        if ((b.loss - a.loss) / a.loss > 0.1) {
            // fitness(a) > fitness(b)
            return false;
        }

        // If a is more complex and the loss difference is insignificant or b is better
        // fitness(a) < fitness(b)
        return true;
    } 
    else 
    {
        // If a is simpler, but a has significantly higher loss
        if ((a.loss - b.loss) / b.loss > 0.1) {
            // fitness(a) < fitness(b)
            return true;
        }

        // If a is simpler, and the loss difference is insignificant or a is better
        // fitness(a) > fitness(b)
        return false;
    }
}
