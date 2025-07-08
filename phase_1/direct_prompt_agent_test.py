"""
Test script for DirectPromptAgent.

This script demonstrates how to use the DirectPromptAgent and verifies its functionality.
"""

import os
import sys
import json
from dotenv import load_dotenv

# Add the parent directory to the path so we can import the workflow_agents package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# Import utils for output file generation
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase_2"))
from utils import format_output_file, save_output_file

# Import the DirectPromptAgent
from workflow_agents.base_agents import DirectPromptAgent

def main():
    """
    Main function to test the DirectPromptAgent.
    """
    # Check if the API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please create a .env file in the tests directory with your OpenAI API key.")
        print("Example: OPENAI_API_KEY=your-api-key-here")
        return

    # Create a DirectPromptAgent with a custom system prompt
    agent = DirectPromptAgent(
        system_prompt="You are a helpful assistant specialized in explaining complex concepts simply."
    )

    # Test the agent with a prompt
    prompt = "Explain the concept of recursion in programming in 3 sentences."

    print(f"Testing DirectPromptAgent with prompt: '{prompt}'")
    print("-" * 50)

    try:
        # Use the respond method instead of run
        response = agent.respond(prompt)

        # Print the response
        print("Response:")
        print(response)
        print("-" * 50)

        # Verify the response is not empty
        assert response, "Response is empty"

        # Generate output file
        output_content = format_output_file(
            prompt=prompt,
            response=response,
            agent_type="DirectPromptAgent",
            knowledge_source="The agent uses its built-in knowledge to respond to the prompt."
        )

        # Save output file
        output_dir = os.path.join(os.path.dirname(__file__), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "direct_prompt_agent_output.txt")
        save_output_file(output_content, output_file)
        print(f"Output saved to {output_file}")

        print("Test completed successfully!")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
