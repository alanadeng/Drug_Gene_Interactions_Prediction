# Transductive Link Prediction for Drug-Gene Interaction Networks
**Alana Deng, Kaitlyn Wade, Kyle Chen, Gen Zhou**

## Bipartite Graph
A bipartite graph is defined as a graph whose vertex set can be partitioned into two disjoint sets. Let \( G = (V, E) \) be a graph with vertex set \( V \) and edge set \( E \). \( G \) is bipartite if there exist two disjoint sets \( U \) and \( W \) such that \( V = U \cup W \) and every edge in \( E \) connects a vertex in \( U \) to a vertex in \( W \). A graph \( G \) is bipartite if and only if there is no cycle of odd length in \( G \).

The notation \( G = (U, W, E) \) is commonly used to denote a bipartite graph with vertex sets \( U \) and \( W \), and edge set \( E \).

## Drug-Gene Interaction Data
Drug-gene interaction data provides a nuanced perspective on how genetic variations can influence an individual's response to specific medications. Biological systems are inherently complex, with intricate relationships between drugs and genes. Capturing and understanding these relationships is crucial for uncovering insights that empower personalized treatments and precision medicine.

Drug-gene interaction data is often modeled as a bipartite graph, where there is a set of vertices \( U \) for different types of drug, and a disjoint vertex set \( V \) for genes. The interactions between them are depicted as edges in set \( E \).

## Transductive Learning
Transductive learning focuses on making predictions for specific, existing data points. In transductive learning, the model aims to infer the labels or relationships of a specific set of instances that are already present in the dataset.

In the context of bipartite link prediction, transductive learning seeks to utilize the information inherent in the current structure of the bipartite graph to make predictions for specific, unobserved links.

## Motivation & Challenge
- Real-world drug-gene interaction data is often incomplete due to experimental constraints. Link prediction addresses this incom
