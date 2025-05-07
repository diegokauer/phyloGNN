from ete3 import Tree


def build_full_tree(taxa_dataframes, use_asv=False, root_name='Life'):
    tree = Tree(name=root_name)
    node_dict = {root_name: tree}

    for taxa_dataframe in taxa_dataframes:
        for _, row in taxa_dataframe.iterrows():
            if use_asv:
                lineage = (row[1:]._append(row[:1])).dropna().to_list()  # Append ASV to the end
            else:
                lineage = row[1:].dropna().to_list()
            current_node = tree

            for taxon in lineage:
                if taxon not in node_dict and taxon != '':
                    new_node = current_node.add_child(name=taxon)
                    node_dict[taxon] = new_node
                if taxon != '':
                    current_node = node_dict[taxon]

    taxa2id = {name: idx for idx, name in enumerate(node_dict.keys())}
    id2taxa = {idx: name for name, idx in taxa2id.items()}

    return taxa2id, id2taxa, tree


def populate_tree_leaves(tree, present_asv, row):
    for asv in present_asv:
        leaf = tree.get_leaves_by_name(asv)[0]
        leaf.add_feature("count", row[asv])
    return tree


def propagate_tree(tree):
    if tree.is_leaf():
        return tree.count
    count = sum([propagate_tree(child) for child in tree.children]) + tree.count
    tree.add_feature("count", count)
    return count


def reset_tree(tree):
    for node in tree.traverse():
        node.add_feature("count", 0)
    return tree