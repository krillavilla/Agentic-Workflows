"""
Test script for AugmentedPromptAgent.

This script demonstrates how to use the AugmentedPromptAgent and verifies its functionality.
"""

import os
import sys
import json
from dotenv import load_dotenv

# Add the parent directory to the path so we can import the workflow_agents package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# Import the AugmentedPromptAgent
from workflow_agents.base_agents import AugmentedPromptAgent

def main():
    """
    Main function to test the AugmentedPromptAgent.
    """
    # Check if the API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please create a .env file in the tests directory with your OpenAI API key.")
        print("Example: OPENAI_API_KEY=your-api-key-here")
        return

    # Create an AugmentedPromptAgent with a custom system prompt, prefix, and suffix
    agent = AugmentedPromptAgent(
        system_prompt="You are a helpful assistant specialized in providing concise answers.",
        prompt_prefix="Please provide a brief answer to the following question: ",
        prompt_suffix=" Answer in no more than 50 words."
    )

    # Test the agent with a prompt
    prompt = "What are the key principles of object-oriented programming?"
    
    print(f"Testing AugmentedPromptAgent with prompt: '{prompt}'")
    print(f"Prefix: '{agent.prompt_prefix}'")
    print(f"Suffix: '{agent.prompt_suffix}'")
    print("-" * 50)
    
    try:
        # Run the agent
        response = agent.run(prompt)
        
        # Print the response
        print("Response:")
        print(response)
        print("-" * 50)
        
        # Verify the response is not empty
        assert response, "Response is empty"
        
        # Create a second agent with different augmentation
        print("Testing with different augmentation:")
        agent2 = AugmentedPromptAgent(
            system_prompt="You are a helpful assistant specialized in providing detailed explanations.",
            prompt_prefix="I need a comprehensive explanation of ",
            prompt_suffix=". Please include examples where appropriate."
        )
        
        # Run the second agent
        response2 = agent2.run(prompt)
        
        # Print the response
        print("Response with different augmentation:")
        print(response2)
        print("-" * 50)
        
        # Verify the response is not empty
        assert response2, "Second response is empty"
        
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()