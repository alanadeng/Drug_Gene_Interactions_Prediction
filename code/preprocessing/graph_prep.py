import numpy as np
import networkx as nx
from node2vec import Node2Vec
import torch
import pandas as pd
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.decomposition import PCA
from scipy.spatial import distance
from gensim.models import Word2Vec
import random


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
                G.add_edge(gene_li[i], drug_li[j])
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


def generate_edges(df):
    # Generate positive & negative examples
    possible_edges = set([(gene, drug) for gene in gene_list(df) for drug in drug_list(df)])
    edges_positive = [(gene, drug) for (gene, drug) in possible_edges if df.loc[gene, drug] == 1]
    labels_positive = [1] * len(edges_positive)
    edges_negative = list(possible_edges - set(edges_positive))  # Negative examples
    labels_negative = [0] * len(edges_negative)

    # Combine positive and negative examples
    #edges = edges_positive + edges_negative
    #labels = labels_positive + labels_negative
    return edges_positive, edges_negative, labels_positive, labels_negative


def edge_features(edge, node_embeddings):
    return np.concatenate([node_embeddings[edge[0]], node_embeddings[edge[1]]])


def negative_sampling(df, node_embeddings, method='random_under', random_state=100, positive_negative_ratio=0.5):
    edges_positive_idx, edges_negative_idx, labels_positive, labels_negative = generate_edges(df)

    edges_positive = np.array([edge_features(edge, node_embeddings) for edge in edges_positive_idx])
    edges_negative = np.array([edge_features(edge, node_embeddings) for edge in edges_negative_idx])

    # Combine positive and negative examples
    edges = np.concatenate((edges_positive, edges_negative), axis=0)
    labels = np.array(labels_positive + labels_negative)

    # Determine the number of negative samples to keep
    num_positives = len(edges_positive)
    num_negatives = int(num_positives / positive_negative_ratio)

    if method == 'random_under':
        # Random under-sampling of negatives
        np.random.seed(random_state)
        neg_indices = np.random.choice(len(edges_negative), num_negatives, replace=False)
        negative_samples = np.array(edges_negative)[neg_indices]
    elif method == 'distance_under':
        # Distance-based under-sampling
        positive_centroid = np.mean(edges_positive, axis=0)
        distances = np.array([distance.euclidean(positive_centroid, neg) for neg in edges_negative])
        neg_indices = np.argsort(distances)[:num_negatives]
        negative_samples = np.array(edges_negative)[neg_indices]
    else:
        raise ValueError("Invalid method. Choose from ['random_under', 'distance_under']")

    # Combine sampled negatives with positives
    sampled_edges = np.concatenate((edges_positive, negative_samples))
    sampled_labels = np.concatenate((labels_positive, [0] * len(negative_samples)))

    return sampled_edges, sampled_labels


def edge_train_test_split(sampled_edges, sampled_labels, test_size=0.2, random_state=100, stratified=True):
    if stratified:
        # Stratified split to maintain the same proportion of labels in both train and test sets
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        for train_index, test_index in sss.split(sampled_edges, sampled_labels):
            edges_train, edges_test = sampled_edges[train_index], sampled_edges[test_index]
            y_train, y_test = sampled_labels[train_index], sampled_labels[test_index]
    elif not stratified:
        # Simple random split
        edges_train, edges_test, y_train, y_test = train_test_split(sampled_edges, sampled_labels, test_size=test_size, random_state=random_state)
    else:
        raise ValueError("Invalid value for 'stratified'. Must be a boolean.")

    return edges_train, edges_test, y_train, y_test


def perform_pca(edge_features, n_components=150):
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(edge_features)
    print("PCA Explained Variance: ", sum(pca.explained_variance_ratio_[:n_components]))
    return reduced_features
