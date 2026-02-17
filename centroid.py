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
ideas = df['use'].fillna("").tolist()
all_embeddings = model.encode(ideas, show_progress_bar=True)

# 4. Pre-calculate Centroids per [Condition + Object]
unique_conditions = df['condition'].unique()
unique_objects = df['object'].unique()
group_centroids = {}

# Create a grouping key to find the "norm" for each condition-object pair
for cond in unique_conditions:
    for obj in unique_objects:
        mask = (df['condition'] == cond) & (df['object'] == obj)
        valid_embeddings = all_embeddings[mask]
        
        if len(valid_embeddings) > 0:
            # Centroid represents the 'average response' for this specific condition and object
            group_centroids[(cond, obj)] = np.mean(valid_embeddings, axis=0)


# 5. Compute Semantic Distance (Relative to the Condition-Object Norm)
distance_results = []

for idx, current_embedding in enumerate(all_embeddings):
    row = df.iloc[idx]
    if not ideas[idx]:
        distance_results.append(np.nan)
        continue
    
    # Identify the specific centroid for this row's condition and object
    centroid = group_centroids.get((row['condition'], row['object']))
    
    if centroid is not None:
        dist = cosine(current_embedding, centroid)
        distance_results.append(dist)
    else:
        distance_results.append(np.nan)

df['semantic_distance'] = distance_results

# 6. Save results
df.to_csv('data/centroids_updated.csv', index=False)


# Print summary stats to check the comparison
print("\n--- Semantic Distance Summary Statistics per Condition ---")
# Mean and 95% confidence interval of distances per condition-object pair
summary_stats = df.groupby('condition')['semantic_distance'].agg(['mean', 'std', 'count']).reset_index()
print(summary_stats)
