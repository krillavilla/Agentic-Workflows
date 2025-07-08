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

# Import utils for output file generation
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase_2"))
from utils import format_output_file, save_output_file

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
        # Use the respond method instead of run
        response = agent.respond(prompt)

        # Print the response
        print("Response:")
        print(response)
        print("-" * 50)

        # Verify the response is not empty
        assert response, "Response is empty"

        # Generate output file
        persona_effect = f"The agent augmented the prompt with prefix: '{agent.prompt_prefix}' and suffix: '{agent.prompt_suffix}' to get a concise answer."
        output_content = format_output_file(
            prompt=prompt,
            response=response,
            agent_type="AugmentedPromptAgent",
            persona_effect=persona_effect
        )

        # Save output file
        output_dir = os.path.join(os.path.dirname(__file__), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "augmented_prompt_agent_output_concise.txt")
        save_output_file(output_content, output_file)
        print(f"Output saved to {output_file}")

        # Create a second agent with different augmentation
        print("Testing with different augmentation:")
        agent2 = AugmentedPromptAgent(
            system_prompt="You are a helpful assistant specialized in providing detailed explanations.",
            prompt_prefix="I need a comprehensive explanation of ",
            prompt_suffix=". Please include examples where appropriate."
        )

        # Use the respond method instead of run
        response2 = agent2.respond(prompt)

        # Print the response
        print("Response with different augmentation:")
        print(response2)
        print("-" * 50)

        # Verify the response is not empty
        assert response2, "Second response is empty"

        # Generate output file for second agent
        persona_effect2 = f"The agent augmented the prompt with prefix: '{agent2.prompt_prefix}' and suffix: '{agent2.prompt_suffix}' to get a detailed explanation."
        output_content2 = format_output_file(
            prompt=prompt,
            response=response2,
            agent_type="AugmentedPromptAgent",
            persona_effect=persona_effect2
        )

        # Save output file for second agent
        output_file2 = os.path.join(output_dir, "augmented_prompt_agent_output_detailed.txt")
        save_output_file(output_content2, output_file2)
        print(f"Output saved to {output_file2}")

        print("Test completed successfully!")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
