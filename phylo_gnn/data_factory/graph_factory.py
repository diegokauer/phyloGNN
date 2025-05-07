import os

import torch
from ete3 import Tree
import networkx as nx
import pandas as pd
import numpy as np
from torch_geometric.data import Data

from phylo_gnn.utils.tree import build_full_tree, populate_tree_leaves, propagate_tree, reset_tree
from phylo_gnn.utils.graph import build_graph_from_tree, connect_sister_leaves, make_symmetric, adj_matrix_to_coo_matrix
from phylo_gnn.data_factory.abstract_factory import AbstractDataFactory


class GraphDataFactory(AbstractDataFactory):
    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)

        # Information about data_set
        self.node_n = 0

    def get_global_batch(self, use_asv):
        taxa2id, id2taxa, global_tree = build_full_tree(
            taxa_dataframes=[self.taxa_data_bacteria, self.taxa_data_fungi],
            use_asv=use_asv,
            root_name="Life"
        )
        self.node_n = len(id2taxa)

        batch = {
            'global_id2taxa': id2taxa,
            'global_taxa2id': taxa2id,
            'global_tree': global_tree,
        }
        self.id2taxa = id2taxa
        self.taxa2id = taxa2id

        return batch

    def construct_dataset(self, df, graph_transformer=None, use_asv=False, taxa_dataframes=None):
        if taxa_dataframes is None:
            taxa_dataframes = [self.taxa_data_bacteria, self.taxa_data_fungi]

        dataset = []
        batch = self.get_global_batch(use_asv)
        asv_cols = [col for col in df.columns if 'ASV' in col]

        for idx, row in df.iterrows():
            present_asv = [asv for asv in asv_cols if row[asv] > 0]

            sub_taxa_dfs = []
            for taxa_df in taxa_dataframes:
                sub_taxa_dfs.append(taxa_df[taxa_df.ASV.isin(present_asv)])

            sample_taxa2id, sample_id2taxa, tree = build_full_tree(
                taxa_dataframes=sub_taxa_dfs,
                use_asv=use_asv,
                root_name="Life"
            )

            graph = build_graph_from_tree(tree)
            adj_matrix = nx.adjacency_matrix(graph).todense()
            adj_matrix = make_symmetric(adj_matrix)
            coo_matrix = adj_matrix_to_coo_matrix(adj_matrix)

            tree = reset_tree(tree)
            taxa = pd.concat(sub_taxa_dfs, axis=0)
            taxa["id"] = taxa.apply(lambda x: '/'.join(x[taxa.columns]).strip('/').split('/')[-1], axis=1)

            for node_name, sub_df in taxa.groupby("id"):
                leaf = tree.search_nodes(name=node_name)[0]
                asv = sub_df.ASV
                leaf.add_feature("count", sum(row[asv]))
                # print(node_name, sum(row[asv]))

            propagate_tree(tree)

            batch['sample_tree'] = tree
            batch['sample_id2taxa'] = sample_id2taxa
            batch['sample_taxa2id'] = sample_taxa2id
            batch['coo_matrix'] = coo_matrix

            datapoint = self.get_graph_features(row, batch, graph_transformer)
            assert datapoint.validate(), f'row {idx} is not valid'
            dataset.append(datapoint)

        return dataset

    def get_graph_features(self, row, batch, graph_transformer):
        global_taxa2id = batch["global_taxa2id"]
        sample_id2taxa = batch["sample_id2taxa"]
        sample_taxa2id = batch["sample_taxa2id"]
        sample_tree = batch["sample_tree"]
        coo_matrix = batch["coo_matrix"]

        node_features = np.zeros((len(sample_id2taxa), 3))
        edge_features = np.ones((coo_matrix.shape[1], 1))

        for node in sample_tree.traverse():
            idx = sample_taxa2id[node.name]
            node_features[idx, 0] = node.count
            node_features[idx, 1] = global_taxa2id[node.name]
            node_features[idx, 2] = int(node.get_distance(sample_tree))

        # Normalize to get relative abundance
        node_features[:, 0] /= node_features[:, 0].max()
        # print(row.index)

        data_point = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=torch.tensor(coo_matrix),
            edge_attr=torch.tensor(edge_features, dtype=torch.float32),
            y=torch.tensor(row["Deterioration"], dtype=torch.float32),
            graph_attr=torch.tensor(
                [row[[
                    "Sex_M",
                    "age_group_5 to 8",
                    "age_group_Over 8",
                    "age_group_Under 5",
                    'zscore_label_Normal',
                    'zscore_label_Thin',
                    "Chao1_16S",
                    "Shannon_16S",
                    "Simpson_16S",
                    "Pielou_16S",
                ]]],
                dtype=torch.float32)
        )

        if graph_transformer:
            data_point = graph_transformer(data_point)

        return data_point
