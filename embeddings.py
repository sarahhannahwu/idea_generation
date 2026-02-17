import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

# 1. Load data
file_path = 'data/evaluated_compliant_ideas.csv' 
df = pd.read_csv(file_path)

# 2. Initialize Model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Generate Embeddings
ideas = df['use'].fillna("").tolist()
all_embeddings = model.encode(ideas, show_progress_bar=True)

# Append embeddings to dataframe for k-means clustering
embedding_cols = [f'embedding_{i}' for i in range(all_embeddings.shape[1])]
embeddings_df = pd.DataFrame(all_embeddings, columns=embedding_cols)
df = pd.concat([df, embeddings_df], axis=1)

# Save the file as a csv
df.to_csv('data/evaluated_compliant_ideas_with_embeddings.csv', index=False)
