// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "genetic/recombination/recombiner/default.hpp"

#include "util/rng.hpp"
#include "genetic/common/expression_picker.hpp"
#include "util/precision.hpp"

#include <tuple>
#include <cassert>
#include <map>

namespace qsr {

std::tuple<Expression, Expression> DefaultRecombiner::recombine(Expression e1, Expression e2) noexcept {
    if (((thread_local_rng() % RAND_MAX) / RAND_MAX) > (double)crossover_probability) {
        return std::make_tuple(e1, e2);
    }

    // 1. Get original weights from both expressions
    const std::vector<double> weights1 = e1.weights;
    const std::vector<double> weights2 = e2.weights;

    // 2. Create working copies of the expressions.
    // All subsequent operations are performed on these copies.
    Expression new_e1 = e1;
    Expression new_e2 = e2;

    // Data structures to hold pointers to all nodes and their parents.
    std::vector<Expression*> nodes1, nodes2;
    std::map<Expression*, Expression*> parent_map1, parent_map2;

    // 3. Helper lambda to traverse a tree and collect node/parent pointers.
    // This replaces the recursive `pick` logic with a single setup pass.
    auto collect_nodes = [](Expression* node, Expression* parent,
                             std::vector<Expression*>& nodes,
                             std::map<Expression*, Expression*>& parent_map) {

        std::function<void(Expression*, Expression*)> traverse =
            [&](Expression* current_node, Expression* current_parent) {

            nodes.push_back(current_node);
            parent_map[current_node] = current_parent;

            for (auto& operand : current_node->operands) {
                traverse(&operand, current_node);
            }
        };

        traverse(node, parent);
    };

    // Populate the node and parent maps for both trees.
    collect_nodes(&new_e1, nullptr, nodes1, parent_map1);
    collect_nodes(&new_e2, nullptr, nodes2, parent_map2);

    // 4. Randomly select one node from each tree to act as the crossover point.
    std::uniform_int_distribution<size_t> dist1(0, nodes1.size() - 1);
    std::uniform_int_distribution<size_t> dist2(0, nodes2.size() - 1);

    Expression* sub1_ptr = nodes1[dist1(thread_local_rng)];
    Expression* sub2_ptr = nodes2[dist2(thread_local_rng)];

    // 5. Swap the contents of the chosen subtrees.
    // std::swap is very efficient; it moves internal data rather than deep-copying.
    std::swap(*sub1_ptr, *sub2_ptr);

    // 6. Helper lambda to update metrics for a single node based on its children.
    // This logic is equivalent to the work done by the Expression constructors.
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

    // 7. Update metrics upwards from the swapped nodes to the root.
    // This targeted update replaces the slow, full-tree `reorganize` function.
    Expression* current = parent_map1[sub1_ptr];
    while (current != nullptr) {
        update_node_metrics(current);
        current = parent_map1[current];
    }

    current = parent_map2[sub2_ptr];
    while (current != nullptr) {
        update_node_metrics(current);
        current = parent_map2[current];
    }

    // Final update for the root nodes themselves, in case they were the parents.
    update_node_metrics(&new_e1);
    update_node_metrics(&new_e2);

    // 8. Depth check: If new trees are too deep, revert to the originals.
    if (new_e1.depth > max_depth || new_e2.depth > max_depth) {
        return std::make_tuple(e1, e2);
    }

    // 9. If both trees have weights, apply whole arithmetic crossover to weights
    if (!weights1.empty() && !weights2.empty()) {
        assert(weights1.size() == weights2.size());
        new_e1.weights.resize(weights1.size());
        new_e2.weights.resize(weights2.size());
        for (int i = 0; i < weights1.size(); i++) {
            double alpha = (thread_local_rng() % RAND_MAX) / RAND_MAX;
            double w1 = alpha * weights1[i] + (1.0 - alpha) * weights2[i];
            double w2 = alpha * weights2[i] + (1.0 - alpha) * weights1[i];
            new_e1.weights[i] = w1;
            new_e2.weights[i] = w2;
        }
    }

    // 10. If one tree has weights and the other does not, copy the weights from
    // the tree which has weights
    if (!weights1.empty() && weights2.empty()) {
        new_e1.weights = weights1;
        new_e2.weights = weights1;
    }
    if (weights1.empty() && !weights2.empty()) {
        new_e1.weights = weights2;
        new_e2.weights = weights2;
    }

    // 9. Return the newly created expressions.
    return std::make_tuple(std::move(new_e1), std::move(new_e2));
}

}