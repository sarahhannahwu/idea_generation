from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import random

# --- Configuration ---
CSV_PATH = "/data/iriss_trial_data.csv"  # your CSV file with a column "use"

# --- Load stimuli ---
df = pd.read_csv(CSV_PATH)
if "use" not in df.columns:
    raise ValueError("CSV must contain a 'use' column")

# --- Initialize app ---
app = FastAPI(title="Creativity Stimulus Sampler")

# Allow CORS so Qualtrics can fetch data directly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify your Qualtrics domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Endpoint: get N random uses ---
@app.get("/sample")
def get_random_uses(n: int = Query(10, ge=1, le=100)):
    """Return n randomly sampled uses from the CSV with their conditions."""
    # Get random indices using random.sample
    available_indices = list(range(len(df)))
    sampled_indices = random.sample(available_indices, min(n, len(df)))
    
    # Get the rows at those indices and return both use and condition
    sample_data = [
        {
            "use": df.iloc[i]["use"],
            "condition": df.iloc[i]["condition"],
            "ResponseId": df.iloc[i]["ResponseId"]
        }
        for i in sampled_indices
    ]
    
    return {"uses": sample_data}

# --- Endpoint: get stratified sample ---
@app.get("/stratified_sample")
def get_stratified_sample(n_per_category: int = Query(5, ge=1, le=50)):
    """Return n randomly sampled uses from each object category with conditions."""
    result = {}
    
    # Get unique object categories
    categories = df["object"].unique()
    
    for category in categories:
        # Filter data for this category
        category_data = df[df["object"] == category]
        
        # Use random.sample to get random indices from this category
        available_indices = list(category_data.index)
        n_samples = min(n_per_category, len(category_data))
        sampled_indices = random.sample(available_indices, n_samples)
        
        # Return both use and condition for each sample
        sample_data = [
            {
                "use": df.loc[i, "use"],
                "condition": df.loc[i, "condition"],
                "object": df.loc[i, "object"],
                "ResponseId": df.loc[i, "ResponseId"]
            }
            for i in sampled_indices
        ]
        
        result[category] = sample_data
    
    # Add summary information
    result["summary"] = {
        "categories_found": len(categories),
        "samples_per_category": n_per_category,
        "total_samples": sum(len(uses) for uses in result.values() if isinstance(uses, list))
    }
    
    return result
