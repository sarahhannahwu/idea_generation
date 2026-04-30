import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

# 1. Load data
file_path = 'data/evaluated_compliant_ideas.csv' 
df = pd.read_csv(file_path)

# 1.5 Sample one idea per submitter-object combination to remove dependence
df = (
    df.groupby(["submitter_id", "object"], group_keys=False)
      .sample(n=1, random_state=42)
      .reset_index(drop=True)
)

# 2. Initialize the Open-Source Model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Generate Embeddings
ideas = df['use'].fillna("").tolist()
all_embeddings = model.encode(ideas, show_progress_bar=True)

# 4. Calculate Similarity: each idea vs. centroid of OTHER participants
#    within the same condition-object combination
results = []

for idx, current_embedding in enumerate(all_embeddings):
    if not ideas[idx]:
        results.append(np.nan)
        continue

    current_row = df.iloc[idx]
    current_submitter = current_row['submitter_id']
    current_condition = current_row['condition']
    current_object = current_row['object']

    mask = (
        (df['condition'] == current_condition) &
        (df['object'] == current_object) &
        (df['submitter_id'] != current_submitter)
    )

    other_embeddings = all_embeddings[mask]

    if len(other_embeddings) > 0:
        average_embedding = np.mean(other_embeddings, axis=0)
        similarity = 1 - cosine(current_embedding, average_embedding)
        results.append(similarity)
    else:
        results.append(np.nan)


# Save results
df['similarity_to_cond_obj'] = results
df.to_csv('data/doshi_centroid_analysis.csv', index=False)

# Print summary statistics by condition
summary = df.groupby('condition')['similarity_to_cond_obj'].agg(['mean', 'std', 'count'])
print(summary)