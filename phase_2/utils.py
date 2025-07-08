"""
Utility functions for agentic workflow.

This module contains utility functions used by the agents in the workflow.
"""

import os
import json
import numpy as np
from openai import OpenAI

# Initialize the OpenAI client with Vocareum configuration
client = OpenAI(
    base_url="https://openai.vocareum.com/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)

def get_embedding(text, model="text-embedding-3-large"):
    """
    Get an embedding for the given text using the specified model.

    Args:
        text (str): The text to get an embedding for
        model (str): The model to use for the embedding

    Returns:
        list: The embedding vector
    """
    try:
        response = client.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {str(e)}")
        # Return a zero vector as fallback
        return [0.0] * 1536  # text-embedding-3-large has 1536 dimensions

def cosine_similarity(vec1, vec2):
    """
    Calculate the cosine similarity between two vectors.

    Args:
        vec1 (list): The first vector
        vec2 (list): The second vector

    Returns:
        float: The cosine similarity between the vectors
    """
    # Convert to numpy arrays
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    # Calculate cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    # Handle zero vectors
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    
    return dot_product / (norm_vec1 * norm_vec2)

def find_best_match(query_embedding, candidate_embeddings, candidates):
    """
    Find the best match for a query embedding among a list of candidate embeddings.

    Args:
        query_embedding (list): The query embedding
        candidate_embeddings (list): List of candidate embeddings
        candidates (list): List of candidates corresponding to the embeddings

    Returns:
        tuple: The best matching candidate and its similarity score
    """
    if not candidate_embeddings or not candidates:
        return None, 0.0
    
    # Calculate similarities
    similarities = [cosine_similarity(query_embedding, emb) for emb in candidate_embeddings]
    
    # Find the best match
    best_idx = np.argmax(similarities)
    best_similarity = similarities[best_idx]
    best_match = candidates[best_idx]
    
    return best_match, best_similarity

def format_output_file(prompt, response, agent_type, knowledge_source=None, persona_effect=None):
    """
    Format the output for a test script.

    Args:
        prompt (str): The prompt used
        response (str): The agent's response
        agent_type (str): The type of agent
        knowledge_source (str, optional): The source of knowledge for KnowledgeAugmentedPromptAgent
        persona_effect (str, optional): Comment on persona/knowledge effect for AugmentedPromptAgent

    Returns:
        str: The formatted output
    """
    output = f"PROMPT:\n{prompt}\n\n"
    output += f"RESPONSE:\n{response}\n\n"
    
    if agent_type == "DirectPromptAgent" and knowledge_source:
        output += f"SOURCE OF KNOWLEDGE: {knowledge_source}\n"
    elif agent_type == "AugmentedPromptAgent" and persona_effect:
        output += f"PERSONA/KNOWLEDGE EFFECT: {persona_effect}\n"
    elif agent_type == "KnowledgeAugmentedPromptAgent":
        output += "KNOWLEDGE WAS USED: The agent augmented the prompt with specific knowledge.\n"
    
    return output

def save_output_file(output, filename):
    """
    Save the output to a file.

    Args:
        output (str): The output to save
        filename (str): The name of the file to save to

    Returns:
        bool: True if the file was saved successfully, False otherwise
    """
    try:
        with open(filename, 'w') as f:
            f.write(output)
        return True
    except Exception as e:
        print(f"Error saving output file: {str(e)}")
        return False

def parse_json_response(response_text):
    """
    Parse a JSON response from an agent.

    Args:
        response_text (str): The response text to parse

    Returns:
        dict: The parsed JSON, or None if parsing failed
    """
    try:
        # Try to parse the response as JSON
        return json.loads(response_text)
    except json.JSONDecodeError:
        # If the response is not valid JSON, try to extract JSON from the text
        try:
            # Look for JSON-like structure in the text
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx+1]
                return json.loads(json_str)
            
            # Try to find JSON array
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']')
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx+1]
                return json.loads(json_str)
            
            return None
        except:
            return None