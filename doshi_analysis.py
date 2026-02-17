import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

# 1. Load data
file_path = 'data/evaluated_compliant_ideas.csv' 
df = pd.read_csv(file_path)

# 2. Initialize the Open-Source Model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Generate Embeddings
ideas = df['use'].fillna("").tolist()
all_embeddings = model.encode(ideas, show_progress_bar=True)

# 4. Calculate Similarity (Leave-One-Out per Condition AND Object)
results = []

for idx, current_embedding in enumerate(all_embeddings):
    if not ideas[idx]:
        results.append(np.nan)
        continue
    
    current_row = df.iloc[idx]
    current_cond = current_row['condition']
    current_obj = current_row['object']
    
    # CORRECTED MASK: same condition AND same object, excluding self
    mask = (df['condition'] == current_cond) & \
           (df['object'] == current_obj) & \
           (df.index != idx)
           
    cond_obj_embeddings = all_embeddings[mask]
    
    if len(cond_obj_embeddings) > 0:
        # Centroid of everyone ELSE in this condition for this specific object
        average_embedding = np.mean(cond_obj_embeddings, axis=0)
        
        # 1 - cosine distance = cosine similarity
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