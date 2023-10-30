import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
import torch_geometric
from sklearn.model_selection import train_test_split


def gene_list(df):
    return df.index.tolist()


def drug_list(df):
    return df.columns.tolist()


def gene_cnt(df):
    return len(df.index.tolist())


def drug_cnt(df):
    return len(df.columns.tolist())


def nx_drug_gene_bipartite(df):
    G = nx.Graph() # initialize a networkx graph

    gene_li = gene_list(df)
    drug_li = drug_list(df)
    num_gene = gene_cnt(df)
    num_drug = drug_cnt(df)

    # Add nodes to the bipartite graph
    G.add_nodes_from(gene_li, bipartite=0)
    G.add_nodes_from(drug_li, bipartite=1)

    # Add edges to the graph
    for i in range(num_gene):
        for j in range(num_drug):
            if df.iloc[i, j] == 1:
                G.add_edge(gene_li, drug_li)
    return G

    
def pyg_drug_gene_bipartite(df):
    gene_li = gene_list(df)
    drug_li = drug_list(df)
    num_gene = gene_cnt(df)
    num_drug = drug_cnt(df)

    # Generate edge indices
    edge_indices = []
    for i in range(num_gene):
        for j in range(num_drug):
            if df.iloc[i, j] == 1:
                edge_indices.append((i, num_gene + j))

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    # Creating a PyG data object
    data = Data(edge_index=edge_index)
    
    return data


def node2vec_embedding(G, dimensions=128, walk_length=15, num_walks=300, workers=4, window=10, min_count=1, batch_words=4):
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers)
    model = node2vec.fit(window=window, min_count=min_count, batch_words=batch_words)

    # Get node embeddings
    node_embeddings = {node: model.wv[node] for node in G.nodes()}
    return node_embeddings


def deepwalk_embedding(G, dimensions=128, walk_length=15, num_walks=300, workers=4, window=10, min_count=1, batch_words=4):
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers, p=1, q=1)
    model = node2vec.fit(window=window, min_count=min_count, batch_words=batch_words)

    # Get node embeddings
    node_embeddings = {node: model.wv[node] for node in G.nodes()}
    return node_embeddings


def generate_edges(df, node_embeddings):
    # Generate positive & negative examples
    possible_edges = set([(gene, drug) for gene in gene_list(df) for drug in drug_list(df)])
    edges_positive = [(gene, drug) for (gene, drug) in possible_edges and df.loc(gene, drug)==1 ]
    labels_positive = [1] * len(edges_positive)
    edges_negative = list(possible_edges - set(edges_positive))  # Negative examples
    labels_negative = [0] * len(edges_negative)

    # Combine positive and negative examples
    edges = edges_positive + edges_negative
    labels = labels_positive + labels_negative
    return edges, labels


def edge_features(edge):
    return np.concatenate([node_embeddings[edge[0]], node_embeddings[edge[1]]])


def edge_train_test_split(edges, labels, method=['stratified','negative','down'],random_state, test_size):
    if method=='stratified':
        edges_train, edges_test, y_train, y_test = train_test_split(edges, labels, test_size=test_size, random_state=random_state, stratify=labels)
    elif method=='negative':
        ret = 0
    elif method=='down':
        ret = 0
    return edges_train, edges_test, y_train, y_test

#file_path = "./data/preprocessed_34_10.tsv"
#df = pd.read_csv(file_path, sep='\t', index_col=0)