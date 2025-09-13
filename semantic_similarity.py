# Use a sentence embedding model to find the semantic similarity of ideas within each experimental condition

import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import os

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define a function to compute semantic similarity
def compute_semantic_similarity(texts):
    embeddings = model.encode(texts, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embeddings, embeddings)
    return cosine_scores
