import sys
import numpy as np
import pandas as pd
import networkx as nx
from torch_geometric.data import Data
import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm
import random
from sklearn.model_selection import train_test_split

if len(sys.argv) != 2:
    print("Usage: python script.py <file_path>")
    sys.exit(1)

file_path = sys.argv[1]
print(f"Provided file path: {file_path}")

df = pd.read_csv(file_path, delimiter='\t', index_col=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
B = nx.Graph()
genes = df.index.tolist()
drugs = df.columns.tolist()
B.add_nodes_from(genes, bipartite=0)
B.add_nodes_from(drugs, bipartite=1)

for gene in genes:
    for drug in drugs:
        if df.at[gene, drug] == 1:
            B.add_edge(gene, drug)
if len(B.edges) == 0:
    raise ValueError("No edges added to the graph. Please check your adjacency matrix.")

num_genes = len(genes)
num_drugs = len(drugs)

gene_features = torch.eye(num_genes)
drug_features = torch.eye(num_drugs)

if num_genes > num_drugs:
    padding = torch.zeros(num_drugs, num_genes - num_drugs)
    drug_features = torch.cat([drug_features, padding], dim=1)
elif num_drugs > num_genes:
    padding = torch.zeros(num_genes, num_drugs - num_genes)
    gene_features = torch.cat([gene_features, padding], dim=1)

node_features = torch.cat([gene_features, drug_features], dim=0)
node_to_idx = {node: idx for idx, node in enumerate(B.nodes())}
edges = [(node_to_idx[u], node_to_idx[v]) for u, v in B.edges()]
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
edge_index = torch.cat([edge_index, edge_index.flip([0])], dim=1)

from torch_geometric.utils import from_networkx
data = from_networkx(B)
data.x = node_features
data = Data(x=node_features, edge_index=edge_index).to(device)

def split_edges(edge_index, test_ratio=0.2):
    num_edges = edge_index.size(1)
    perm = torch.randperm(num_edges)
    edge_index = edge_index[:, perm]
    num_test = int(num_edges * test_ratio)
    test_edge_index = edge_index[:, :num_test]
    train_edge_index = edge_index[:, num_test:]

    return train_edge_index, test_edge_index

train_edge_index, test_edge_index = split_edges(data.edge_index)

def create_pos_neg_edges(edge_index, num_nodes, percent=0.1):
    edges_set = set([tuple(edge) for edge in edge_index.t().tolist()])
    all_pos_edges = list(edges_set)
    num_pos_samples = int(len(all_pos_edges) * percent)
    pos_edges_sample = random.sample(all_pos_edges, num_pos_samples)
    all_possible_edges = set((i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j)
    non_existing_edges = all_possible_edges - edges_set
    num_neg_samples = num_pos_samples
    neg_edges_sample = random.sample(list(non_existing_edges), num_neg_samples)
    pos_edge_index = torch.tensor(pos_edges_sample, dtype=torch.long).t()
    neg_edge_index = torch.tensor(neg_edges_sample, dtype=torch.long).t()

    return pos_edge_index, neg_edge_index


class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.edge_predictor = torch.nn.Linear(2 * hidden_channels, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

model = GCN(num_features=data.num_features, hidden_channels=16).to(device)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    optimizer.zero_grad()
    node_embeddings = model(data.x.to(device), train_edge_index.to(device))
    pos_edge_index, neg_edge_index = create_pos_neg_edges(train_edge_index.to(device), num_nodes=data.num_nodes)
    pos_edge_features = torch.cat([node_embeddings[pos_edge_index[0]], node_embeddings[pos_edge_index[1]]], dim=1)
    neg_edge_features = torch.cat([node_embeddings[neg_edge_index[0]], node_embeddings[neg_edge_index[1]]], dim=1)
    all_edge_features = torch.cat([pos_edge_features, neg_edge_features], dim=0)
    pos_labels = torch.ones(pos_edge_features.size(0), dtype=torch.float).to(device)
    neg_labels = torch.zeros(neg_edge_features.size(0), dtype=torch.float).to(device)
    all_labels = torch.cat([pos_labels, neg_labels], dim=0)
    edge_scores = model.edge_predictor(all_edge_features).squeeze(1)
    edge_scores = torch.sigmoid(edge_scores)
    loss = criterion(edge_scores, all_labels)
    loss.backward()
    optimizer.step()
    return loss

def create_neg_edges(edge_index, num_nodes, num_neg_samples):
    edges_set = set([tuple(edge) for edge in edge_index.t().tolist()])
    all_possible_edges = set((i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j)
    non_existing_edges = all_possible_edges - edges_set
    neg_edges_sample = random.sample(list(non_existing_edges), num_neg_samples)

    neg_edge_index = torch.tensor(neg_edges_sample, dtype=torch.long).t()
    return neg_edge_index

def test(test_edge_index, data, num_nodes):
    model.eval()
    with torch.no_grad():
        node_embeddings = model(data.x.to(device), test_edge_index.to(device))
        test_pos_edge_index = test_edge_index.to(device)
        test_neg_edge_index = create_neg_edges(test_edge_index.to(device), num_nodes, test_pos_edge_index.size(1)).to(device)
        test_pos_edge_features = torch.cat([node_embeddings[test_pos_edge_index[0]], node_embeddings[test_pos_edge_index[1]]], dim=1)
        test_neg_edge_features = torch.cat([node_embeddings[test_neg_edge_index[0]], node_embeddings[test_neg_edge_index[1]]], dim=1)
        all_test_edge_features = torch.cat([test_pos_edge_features, test_neg_edge_features], dim=0)
        test_pos_labels = torch.ones(test_pos_edge_features.size(0), dtype=torch.float).to(device)
        test_neg_labels = torch.zeros(test_neg_edge_features.size(0), dtype=torch.float).to(device)
        all_test_labels = torch.cat([test_pos_labels, test_neg_labels], dim=0)
        test_edge_scores = model.edge_predictor(all_test_edge_features).squeeze(1)
        test_edge_scores = torch.sigmoid(test_edge_scores)
        test_loss = criterion(test_edge_scores, all_test_labels)

    return test_loss

num_nodes = len(genes) + len(drugs)
train_losses = []
test_losses = []
for epoch in tqdm(range(200)):
    train_loss = train()
    test_loss = test(test_edge_index, data, num_nodes)
    train_losses.append(train_loss.item())
    test_losses.append(test_loss.item())

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
def evaluate_model(model, pos_edge_index, neg_edge_index, data):
    model.eval()
    with torch.no_grad():
        node_embeddings = model(data.x.to(device), data.edge_index.to(device))
        all_edges = torch.cat([pos_edge_index.to(device), neg_edge_index.to(device)], dim=-1)
        edge_scores = torch.sigmoid(model.edge_predictor(
            torch.cat([node_embeddings[all_edges[0]],
                       node_embeddings[all_edges[1]]], dim=1)).squeeze(1))
        actual_labels = torch.cat([torch.ones(pos_edge_index.size(1)),
                                   torch.zeros(neg_edge_index.size(1))], dim=0)
        roc_score = roc_auc_score(actual_labels.cpu(), edge_scores.cpu())
        precision, recall, _ = precision_recall_curve(actual_labels.cpu(), edge_scores.cpu())
        pr_score = auc(recall, precision)

    return roc_score, pr_score

train_neg_edge_index = create_neg_edges(train_edge_index, num_nodes, train_edge_index.size(1))
train_roc_score, train_pr_score = evaluate_model(model, train_edge_index, train_neg_edge_index, data)
print(f'Train ROC AUC Score: {train_roc_score}')
print(f'Train Precision-Recall AUC Score: {train_pr_score}')
test_pos_edge_index = test_edge_index
test_neg_edge_index = create_neg_edges(test_edge_index, num_nodes, test_pos_edge_index.size(1))
roc_score, pr_score = evaluate_model(model, test_pos_edge_index, test_neg_edge_index, data)
print(f'Test ROC AUC Score: {roc_score}')
print(f'Test Precision-Recall AUC Score: {pr_score}')

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Losses Over Epochs')
plt.legend()
plt.show()
