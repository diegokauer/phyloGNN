import networkx as nx
import numpy as np


def build_graph_from_tree(tree):
    graph = nx.DiGraph()
    for node in tree.traverse():
        for child in node.children:
            graph.add_edge(node.name, child.name)
    return graph


def connect_sister_leaves(tree, graph):
    for node in tree.traverse():
        # Add edges connecting the leaves of the same parent
        leaves = [child.name for child in node.children if child.is_leaf()]
        for i in range(len(leaves)):
            for j in range(len(leaves)):
                if i != j:
                    graph.add_edge(leaves[i], leaves[j])
    return graph


def make_symmetric(adj_matrix):
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix)):
            if adj_matrix[i, j] == 1:
                adj_matrix[j, i] = 1
    return adj_matrix


def adj_matrix_to_coo_matrix(adj_matrix):
    row_idx, col_idx = np.where(adj_matrix == 1)
    coo_matrix = np.vstack((row_idx, col_idx))
    return coo_matrix
