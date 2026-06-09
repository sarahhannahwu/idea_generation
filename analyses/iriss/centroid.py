import os

import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import entropy
from sentence_transformers import SentenceTransformer
import seaborn as sns
import umap
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import hdbscan
from sklearn.metrics.pairwise import cosine_distances

# 1. Load data
file_path = 'data/evaluated_compliant_ideas.csv' 
df = pd.read_csv(file_path)

# 2. Initialize Model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Generate Embeddings
# Preprocess ideas (e.g., strip whitespace, convert to lowercase) to ensure consistency
ideas = df['use'].fillna("").tolist()
ideas = [idea.strip().lower() for idea in ideas]
all_embeddings = model.encode(ideas, show_progress_bar=True)

# Compute cosine distance between each idea and the centroid of all other ideas in that condition-object pair
df['semantic_distance'] = np.nan  # Initialize a new column for semantic distance
for (condition, obj), group in df.groupby(['condition', 'object']):
    indices = group.index.tolist()
    if len(indices) > 1:  # Ensure there are at least 2 ideas to compute a centroid
        for idx in indices:
            # Compute the centroid of all other ideas in the same condition-object pair
            other_indices = [i for i in indices if i != idx]
            if other_indices:  # Check if there are other ideas to compute the centroid
                centroid = np.mean(all_embeddings[other_indices], axis=0)
                # Compute cosine distance between the idea and the centroid
                distance = cosine(all_embeddings[idx], centroid)
                df.at[idx, 'semantic_distance'] = distance


# Print summary stats to check the comparison
print("\n--- Semantic Distance Summary Statistics per Condition ---")
# Descriptives of distances per condition-object pair
summary_stats = df.groupby('condition')['semantic_distance'].agg(['mean', 'std', 'count']).reset_index()
print(summary_stats)

# Save the file as a csv
output_path = os.path.expanduser('~/Git_Projects/psych252/final-project-sarah-wu/data/centroids.csv')
df.to_csv(output_path, index=False)
