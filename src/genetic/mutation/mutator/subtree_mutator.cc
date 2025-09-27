// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "genetic/mutation/mutator/subtree_mutator.hpp"

#include "util/precision.hpp"
#include "util/rng.hpp"

#include <map>

namespace qsr {

static Config create_generator_config(const Config& base_config, int new_max_depth) {
    Config generator_config = base_config;
    generator_config.max_depth = new_max_depth;
    return generator_config;
}

SubtreeMutator::SubtreeMutator(const Config &config, double mutation_probability, int max_depth_increment)  :
    config(config),
    mutation_probability(mutation_probability),
    max_depth_increment(max_depth_increment),
    expression_generator(create_generator_config(config, max_depth_increment)) {}


Expression SubtreeMutator::mutate(const Expression &expr) noexcept {
    if (((thread_local_rng() % RAND_MAX) / (double)(RAND_MAX)) > mutation_probability) {
        return expr;
    }

    // 2. Create a working copy of the expression.
    Expression new_expr = expr;

    // Data structures to hold pointers to all nodes and their parents.
    std::vector<Expression*> nodes;
    std::map<Expression*, Expression*> parent_map;

    // 3. Helper lambda to traverse the tree and collect node/parent pointers.
    auto collect_nodes = [](Expression* node, Expression* parent,
                             std::vector<Expression*>& node_list,
                             std::map<Expression*, Expression*>& p_map) {

        std::function<void(Expression*, Expression*)> traverse =
            [&](Expression* current_node, Expression* current_parent) {

            node_list.push_back(current_node);
            p_map[current_node] = current_parent;

            for (auto& operand : current_node->operands) {
                traverse(&operand, current_node);
            }
        };

        traverse(node, parent);
    };

    // Populate the node and parent maps for the tree.
    collect_nodes(&new_expr, nullptr, nodes, parent_map);

    // 4. Randomly select one node from the tree to be mutated.
    std::uniform_int_distribution<size_t> node_dist(0, nodes.size() - 1);
    Expression* mutation_point_ptr = nodes[node_dist(thread_local_rng)];

    // 5. Replace the chosen subtree with a newly generated one.
    // The assignment operator handles the replacement of data.
    *mutation_point_ptr = expression_generator.generate(max_depth_increment, false);

    // 6. Helper lambda to update metrics for a single node.
    auto update_node_metrics = [](Expression* node) {
        if (node->operands.empty()) {
            node->num_of_nodes = 1;
            node->depth = 1;
        } else if (node->operands.size() == 1) {
            node->num_of_nodes = 1 + node->operands[0].num_of_nodes;
            node->depth = 1 + node->operands[0].depth;
        } else { // Assuming binary for simplicity
            node->num_of_nodes = 1 + node->operands[0].num_of_nodes + node->operands[1].num_of_nodes;
            node->depth = 1 + std::max(node->operands[0].depth, node->operands[1].depth);
        }
    };

    // 7. Update metrics upwards from the mutated node to the root.
    // This targeted update replaces the slow, full-tree `reorganize` function.
    Expression* current = parent_map[mutation_point_ptr];
    while (current != nullptr) {
        update_node_metrics(current);
        current = parent_map[current];
    }

    // Final update for the root node itself, in case it was the parent.
    update_node_metrics(&new_expr);

    // 8. Depth check: If the new tree is too deep, revert to the original.
    if (new_expr.depth > config.max_depth) {
        return expr;
    }

    // 9. Return the newly created expression.
    return new_expr;
}

}