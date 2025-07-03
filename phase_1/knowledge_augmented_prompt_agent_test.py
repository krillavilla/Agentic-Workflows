"""
Test script for KnowledgeAugmentedPromptAgent.

This script demonstrates how to use the KnowledgeAugmentedPromptAgent and verifies its functionality.
"""

import os
import sys
import json
from dotenv import load_dotenv

# Add the parent directory to the path so we can import the workflow_agents package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# Import the KnowledgeAugmentedPromptAgent
from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent

def main():
    """
    Main function to test the KnowledgeAugmentedPromptAgent.
    """
    # Check if the API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please create a .env file in the tests directory with your OpenAI API key.")
        print("Example: OPENAI_API_KEY=your-api-key-here")
        return

    # Define some knowledge about a fictional company
    company_knowledge = """
    Company: TechInnovate Solutions
    Founded: 2010
    Headquarters: San Francisco, CA
    Employees: 500+
    
    Products:
    1. CloudSync - A cloud storage and synchronization platform
    2. DataAnalyzer - A business intelligence and data analytics tool
    3. SecureConnect - An enterprise security and VPN solution
    
    Recent News:
    - TechInnovate Solutions recently acquired DataViz Inc. to enhance their data visualization capabilities
    - The company is planning to expand to European markets in Q3 2023
    - CloudSync 2.0 is scheduled for release next month with improved performance and new features
    """

    # Create a KnowledgeAugmentedPromptAgent with the company knowledge
    agent = KnowledgeAugmentedPromptAgent(
        system_prompt="You are a helpful assistant representing TechInnovate Solutions. Provide accurate information about the company.",
        knowledge=company_knowledge
    )

    # Test the agent with a prompt
    prompt = "Tell me about your company's products."
    
    print(f"Testing KnowledgeAugmentedPromptAgent with prompt: '{prompt}'")
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
        
        # Test with a different prompt
        prompt2 = "What are your expansion plans?"
        print(f"Testing with different prompt: '{prompt2}'")
        
        response2 = agent.run(prompt2)
        
        # Print the response
        print("Response to second prompt:")
        print(response2)
        print("-" * 50)
        
        # Verify the response is not empty
        assert response2, "Second response is empty"
        
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()