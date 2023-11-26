# Transductive Link Prediction for Drug-Gene Interaction Networks

**Alana Deng, Kaitlyn Wade, Lianghong Chen, Gen Zhou**

## Background
### Bipartite Graph
A bipartite graph is defined as a graph whose vertex set can be partitioned into two disjoint sets. Let $G = (V, E)$ be a graph with vertex set $V$ and edge set $E$. $G$ is bipartite if there exist two disjoint sets $U$ and $W$ such that $V = U \cup W$ and every edge in $E$ connects a vertex in $U$ to a vertex in $W$. A graph $G$ is bipartite if and only if there is no cycle of odd length in $G$.

The notation $G = (U, W, E)$ is commonly used to denote a bipartite graph with vertex sets $U$ and $W$, and edge set $E$.

### Drug-Gene Interaction Data
Drug-gene interaction data provides a nuanced perspective on how genetic variations can influence an individual's response to specific medications. Biological systems are inherently complex, with intricate relationships between drugs and genes. Capturing and understanding these relationships is crucial for uncovering insights that empower personalized treatments and precision medicine.

Drug-gene interaction data is often modeled as a bipartite graph, where there is a set of vertices $U$ for different types of drug, and a disjoint vertex set $V$ for genes. The interactions between them are depicted as edges in set $E$.

### Transductive Learning
Transductive learning focuses on making predictions for specific, existing data points. In transductive learning, the model aims to infer the labels or relationships of a specific set of instances that are already present in the dataset.

In the context of bipartite link prediction, transductive learning seeks to utilize the information inherent in the current structure of the bipartite graph to make predictions for specific, unobserved links.

## Motivation & Challenges
- Real-world drug-gene interaction data is often incomplete due to experimental constraints. Link prediction addresses this incompleteness by inferring potential interactions based on existing drug-gene interaction data.
- Drug-gene interaction networks are usually sparse, with only a small portion of possible interactions observed. This sparsity poses challenges for accurately predicting links between drugs and genes.
- Integration of diverse data sources adds complexity to the prediction task. Link prediction must navigate these heterogeneous sources to generate meaningful insights.

## Dataset
The Drug-Gene Interaction Database (DGIdb, www.dgidb.org) is a web resource that provides information on drug-gene interactions and druggable genes from publications, databases, and other web-based sources. Drug, gene, and interaction data are normalized and merged into conceptual groups.
- This is a relatively large set for limited computational resources. We can use some sampling/preprocessing methods to select some important nodes/edges

## Methods
- Negative Edge Sampling: Random undersampling, Euclidean distance-based undersampling
- Train-Test Split: Stratified train-test split (Ablation)
- Feature Selection: PCA (Ablation)
- Link Prediction: Node Embeddings + Machine Learning Classifiers
    - Node Embedding:
        - Conventional Methods: Node2Vec, DeepWalk
        - GNN-Based Methods: GraphSAGE, GCN, GAT
    - Classification Models: Logistic regression, XGBoost, SVM, Softmax etc.
- Evaluation Metrics: Accuracy, precision, recall, F1-score, AUC-ROC, AUC-PRC

## Directory Layout

The root directory of the repository is `drug_gene_interaction_prediction`. The folder structure of `drug_gene_interaction_prediction` is as follows

<details><summary>drug_gene_interaction_prediction</summary>
	
    drug_gene_interaction_prediction/
    │
    ├── README.md
    │
    ├── data
    │   ├── interactions.tsv #Raw drug-gene interaction data without any preprocessing
    │   ├── preprocessed_34_10.tsv #preprocessed interaction matrix (cutoff values: gene 34, drug 10)
    │   └── preprocessed_42_10.tsv #preprocessed interaction matrix (cutoff values: gene 42, drug 10)
    │
    ├── code
    │   ├── preprocessing.ipynb #data preprocessing, analysis, and visualization
    │   ├── graph_prep.py #some functions to prepare graph for training
    │   ├── pipeline.ipynb # the node-embedding -> feature selection -> train-test-split -> classification -> evaluation pipeline
    │   ├── plot.ipynb # ROC and PRC curves
    │   └── analysis.ipynb # result analysis
    │
    └── res # directory to store results
        
      
</details>


## Getting Started

- You can read `preprocessing.ipynb` as it shows the steps for preprocessing and data analysis.
- The row indices of preprocessed data are genes, and the column indices are drug types (compounds).
- To use GNN-based methods for node embedding, you can use PyG (PyTorch Geometric). You should first create a PyG data object before generating node embedding. Here are some useful documentations:
  - https://pytorch-geometric.readthedocs.io/en/latest/
  - https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#data-objects
- If you are more familiar with Tensorflow, it looks like Tensorflow also supports GNN. You can see https://blog.tensorflow.org/2021/11/introducing-tensorflow-gnn.html

## Usage

- ../codes/requirements.txt: `pip install -r requirements.txt`.
- ../codes/GCN.py: `python GCN.py <.tsv path>`.
