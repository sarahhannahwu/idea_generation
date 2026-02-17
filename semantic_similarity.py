# Use a sentence embedding model to find the semantic similarity of ideas within each experimental condition

import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import scipy.stats as stats


# Load the sentence transformer model
model = SentenceTransformer('nomic-ai/nomic-embed-text-v1', trust_remote_code=True)

# Load the data

data_path = '~/Git_Projects/idea_generation/data/evaluated_compliant_ideas.csv'  # Update with your actual data path
df = pd.read_csv(data_path)

# For each combination of experimental condition, object, and submitter_id, compute the semantic similarity of ideas
results = []
for (condition, obj, submitter_id), group in df.groupby(['condition', 'object', 'submitter_id']):
    ideas = group['use'].tolist()

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
            'submitter_id': submitter_id,
            'similarity': sim
        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save results to CSV
output_path = os.path.expanduser('~/Git_Projects/idea_generation/data/nomic_similarities.csv')  # Update with your desired output path
results_df.to_csv(output_path, index=False)

# Compute mean and standard deviation of similarities for each condition
summary = results_df.groupby('condition')['similarity'].agg(['mean', 'std'])

# Display summary statistics
print(summary)

