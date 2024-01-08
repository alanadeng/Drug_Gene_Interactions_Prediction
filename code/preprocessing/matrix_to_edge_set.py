import pandas as pd

file_path = './data/dgidb/preprocessed_34_10.tsv'
interaction_matrix = pd.read_csv(file_path, sep='\t', index_col=0)


df_transposed = interaction_matrix.transpose()
df_long = df_transposed.stack().reset_index()
df_long.columns = ['Drug', 'Gene', 'Interaction']
save_path = './data/dgidb/positive_edges/'
df_long[df_long['Interaction'] == 1].to_csv(save_path+'drug_gene_positive_edge_set.csv', index=False, header=None)


# Create mappings for drugs and genes
drug_mapping = {drug: f'u{i+1}' for i, drug in enumerate(df_long['Drug'].unique())}
gene_mapping = {gene: f'i{i+1}' for i, gene in enumerate(df_long['Gene'].unique())}

# Apply mappings to the DataFrame
df_long['Drug'] = df_long['Drug'].map(drug_mapping)
df_long['Gene'] = df_long['Gene'].map(gene_mapping)
df_long = df_long[df_long['Interaction'] == 1]

# Save the modified DataFrame
df_long = df_long[df_long['Interaction'] == 1]
df_long.to_csv(save_path+'drug_gene_positive_edge_set_renamed.dat', sep='\t', index=False, header=None)

# Convert mappings to DataFrame and save
drug_mapping = pd.DataFrame(list(drug_mapping.items()), columns=['Original Drug', 'Renamed']).to_csv(save_path+'drug_mapping.csv', index=False)
gene_mapping = pd.DataFrame(list(gene_mapping.items()), columns=['Original Gene', 'Renamed']).to_csv(save_path+'gene_mapping.csv', index=False)


