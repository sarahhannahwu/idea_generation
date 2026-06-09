# Import csv of ideas and sample a fixed number of ideas per condition-object pair
import pandas as pd
import os
import numpy as np
import requests
import json

# Import Qualtrics configuration
try:
    from qualtrics_config import QUALTRICS_API_TOKEN, QUALTRICS_DATACENTER, SURVEY_ID
except ImportError:
    # Fallback to default values if config file doesn't exist
    QUALTRICS_API_TOKEN = "YOUR_API_TOKEN_HERE"
    QUALTRICS_DATACENTER = "your-datacenter"
    SURVEY_ID = "YOUR_SURVEY_ID_HERE"

# Load the data
data_path = os.path.expanduser('~/Git_Projects/idea_generation/data/iriss_trial_data.csv')  # Update with your actual data path
df = pd.read_csv(data_path)

# Define the number of ideas to sample per condition-object pair
n_samples = 5
sampled_ideas = []
# For each combination of experimental condition and object, sample n ideas
for (condition, obj), group in df.groupby(['condition', 'object']):
    sampled_group = group.sample(n=n_samples, random_state=42)  # Set random_state for reproducibility
    sampled_ideas.append(sampled_group)

# Combine all sampled ideas into a single DataFrame
sampled_df = pd.concat(sampled_ideas)

# Save the sampled ideas to a new CSV
output_path = os.path.expanduser('~/Git_Projects/idea_generation/data/sampled_ideas.csv')  # Update with your desired output path
sampled_df.to_csv(output_path, index=False)

# Use the Qualtrics API to create a new survey with the sampled ideas

def create_qualtrics_question(api_token, datacenter, survey_id, question_text, choices):
    """
    Create a multiple choice question in Qualtrics with the provided choices.
    
    Args:
        api_token (str): Your Qualtrics API token
        datacenter (str): Your Qualtrics datacenter
        survey_id (str): The survey ID where you want to add the question
        question_text (str): The text of the question
        choices (list): List of choice texts for the question
    
    Returns:
        dict: API response
    """
    # Construct the API URL
    base_url = f"https://{datacenter}.qualtrics.com/API/v3/survey-definitions/{survey_id}/questions"
    
    # Set up headers
    headers = {
        'X-API-TOKEN': api_token,
        'Content-Type': 'application/json'
    }
    
    # Create choices dictionary for the API
    choices_dict = {}
    for i, choice in enumerate(choices, 1):
        choices_dict[str(i)] = {
            "Display": choice
        }
    
    # Construct the request payload
    payload = {
        "QuestionText": question_text,
        "DataExportTag": "Q1",  # You can customize this
        "QuestionType": "MC",   # Multiple Choice
        "Selector": "SAVR",     # Single Answer Vertical
        "Configuration": {
            "QuestionDescriptionOption": "UseText"
        },
        "QuestionDescription": "Select one option",
        "Choices": choices_dict,
        "ChoiceOrder": list(choices_dict.keys()),
        "Validation": {
            "Settings": {
                "ForceResponse": "ON",
                "Type": "None"
            }
        },
        "Language": []
    }
    
    try:
        response = requests.post(base_url, headers=headers, json=payload)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        return None

# Prepare the sampled ideas for Qualtrics
def prepare_ideas_for_qualtrics(sampled_df, condition_filter=None, object_filter=None):
    """
    Prepare sampled ideas for uploading to Qualtrics.
    
    Args:
        sampled_df (DataFrame): DataFrame containing sampled ideas
        condition_filter (str, optional): Filter by specific condition
        object_filter (str, optional): Filter by specific object
    
    Returns:
        list: List of idea texts formatted for Qualtrics choices
    """
    # Filter data if specified
    filtered_df = sampled_df.copy()
    if condition_filter:
        filtered_df = filtered_df[filtered_df['condition'] == condition_filter]
    if object_filter:
        filtered_df = filtered_df[filtered_df['object'] == object_filter]
    
    # Extract the ideas (assuming the column is named 'use' or similar)
    # Adjust the column name based on your actual data structure
    if 'use' in filtered_df.columns:
        ideas = filtered_df['use'].tolist()
    elif 'idea' in filtered_df.columns:
        ideas = filtered_df['idea'].tolist()
    else:
        # If unsure, print available columns
        print("Available columns:", filtered_df.columns.tolist())
        ideas = filtered_df.iloc[:, -1].tolist()  # Use the last column as default
    
    # Clean and format ideas
    formatted_ideas = []
    for i, idea in enumerate(ideas, 1):
        # Remove any leading/trailing whitespace and ensure it's a string
        clean_idea = str(idea).strip()
        # Optionally, you can add numbering or formatting
        formatted_ideas.append(clean_idea)
    
    return formatted_ideas

# Example usage
if __name__ == "__main__":
    # Load your sampled data
    print("Sampled ideas saved to:", output_path)
    print(f"Total sampled ideas: {len(sampled_df)}")
    print(f"Ideas by condition:")
    print(sampled_df.groupby('condition').size())
    
    # Check if API credentials are configured
    if QUALTRICS_API_TOKEN == "YOUR_API_TOKEN_HERE":
        print("\n" + "="*50)
        print("QUALTRICS API SETUP REQUIRED")
        print("="*50)
        print("To use the Qualtrics API, please update the following variables:")
        print("1. QUALTRICS_API_TOKEN - Your API token from Qualtrics")
        print("2. QUALTRICS_DATACENTER - Your datacenter (e.g., 'sjc1', 'fra1')")
        print("3. SURVEY_ID - The ID of the survey where you want to add questions")
        print("\nYou can find these in your Qualtrics account settings.")
        print("="*50)
    else:
        # Example: Create a question with all sampled ideas
        all_ideas = prepare_ideas_for_qualtrics(sampled_df)
        question_text = "Please select the most creative use for the given object:"
        
        print(f"\nPreparing to create Qualtrics question with {len(all_ideas)} choices...")
        
        # Create the question
        result = create_qualtrics_question(
            api_token=QUALTRICS_API_TOKEN,
            datacenter=QUALTRICS_DATACENTER,
            survey_id=SURVEY_ID,
            question_text=question_text,
            choices=all_ideas
        )
        
        if result:
            print("Question created successfully!")
            print("Response:", json.dumps(result, indent=2))
        else:
            print("Failed to create question.")
    
    # Optional: Create separate questions for each condition-object pair
    print("\n" + "="*30)
    print("CONDITION-OBJECT BREAKDOWN")
    print("="*30)
    for (condition, obj), group in sampled_df.groupby(['condition', 'object']):
        ideas = prepare_ideas_for_qualtrics(sampled_df, condition, obj)
        print(f"{condition} - {obj}: {len(ideas)} ideas")
        # You could create separate questions for each combination if needed
