# Use a sentence embedding model to find the semantic similarity of ideas within each experimental condition

import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import scipy.stats as stats


# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the data

data_path = '~/Desktop/idea_generation/idea_submissions.csv'  # Update with your actual data path
df = pd.read_csv(data_path)

# For each pair of experimental condition and object, compute the semantic similarity of ideas
results = []
for (condition, obj), group in df.groupby(['condition', 'object']):
    ideas = group['use'].tolist()
    if len(ideas) < 2:
        continue  # Skip groups with less than 2 ideas

    # Compute embeddings
    embeddings = model.encode(ideas)

    # Compute cosine similarity matrix
    norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    similarity_matrix = np.dot(norm_embeddings, norm_embeddings.T)

    # Extract upper triangle of the similarity matrix, excluding the diagonal
    upper_tri_indices = np.triu_indices_from(similarity_matrix, k=1)
    similarities = similarity_matrix[upper_tri_indices]

    # Store results
    for sim in similarities:
        results.append({
            'condition': condition,
            'object': obj,
            'similarity': sim
        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save results to CSV
output_path = os.path.expanduser('~/Desktop/idea_generation/semantic_similarities.csv')  # Update with your desired output path
results_df.to_csv(output_path, index=False)

# Compute mean and standard deviation of similarities for each condition
summary = results_df.groupby('condition')['similarity'].agg(['mean', 'std'])

# Display summary statistics
print(summary)

# Save summary statistics to CSV
summary_output_path = os.path.expanduser('~/Desktop/idea_generation/semantic_similarity_summary.csv')  # Update with your desired output path
summary.to_csv(summary_output_path)

# Run a simple ANOVA to see if there are significant differences between conditions
anova_results = stats.f_oneway(*(group['similarity'].values for name, group in results_df.groupby('condition')))
print("ANOVA results:")
print(f"F-statistic: {anova_results.statistic}")
print(f"P-value: {anova_results.pvalue}")
