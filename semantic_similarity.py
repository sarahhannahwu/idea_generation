# Use a sentence embedding model to find the semantic similarity of ideas within each experimental condition

import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import scipy.stats as stats
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Load the sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load the data

data_path = '~/Git_Projects/idea_generation/data/evaluated_compliant_ideas.csv'  # Update with your actual data path
df = pd.read_csv(data_path)

# Compute the pairwise cosine similarity of ideas by condition and object
# Create a new column to track whether the comparison is within the same submitter or not

results = []
for (condition, obj), group in df.groupby(['condition', 'object']):
    ideas = group['use'].tolist()

    # Preprocess ideas (e.g., strip whitespace, convert to lowercase) to ensure consistency
    ideas = [idea.strip().lower() for idea in ideas]

    # Compute embeddings
    embeddings = model.encode(ideas)

    # Compute cosine similarity matrix
    norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    similarity_matrix = np.dot(norm_embeddings, norm_embeddings.T)

    # Extract upper triangle of the similarity matrix, excluding the diagonal
    upper_tri_indices = np.triu_indices_from(similarity_matrix, k=1)
    similarities = similarity_matrix[upper_tri_indices]
    # Track whether the comparisons are within the same submitter or not
    submitters = group['submitter_id'].tolist()
    submitter_comparisons = []
    for i, j in zip(*upper_tri_indices):
        if submitters[i] == submitters[j]:
            submitter_comparisons.append('same_submitter')
        else:
            submitter_comparisons.append('different_submitter')
    # Add a column to the results with the submitter_id for the idea being left out of the centroid calculation (i.e., the idea being compared to the centroid of all other ideas in that condition-object pair)
    submitter_ids = []
    for i, j in zip(*upper_tri_indices):
        submitter_ids.append(submitters[i])

    # Store results
    for sim in similarities:
        results.append({
            'condition': condition,
            'object': obj,
            'similarity': sim,
            'submitter_comparison': submitter_comparisons.pop(0) if submitter_comparisons else None,
            'submitter_id': submitter_ids.pop(0) if submitter_ids else None
        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save results to CSV
output_path = os.path.expanduser('~/Git_Projects/psych252/final-project-sarah-wu/data/pairwise_similarities.csv')  # Update with your desired output path
results_df.to_csv(output_path, index=False)

# Compute mean and standard deviation of similarities for each condition
summary = results_df.groupby('condition')['similarity'].agg(['mean', 'std'])

# Display summary statistics
print(summary)

