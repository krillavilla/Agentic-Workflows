"""
Test script for EvaluationAgent.

This script demonstrates how to use the EvaluationAgent and verifies its functionality.
"""

import os
import sys
import json
from dotenv import load_dotenv

# Add the parent directory to the path so we can import the workflow_agents package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# Import the EvaluationAgent
from workflow_agents.base_agents import EvaluationAgent

def main():
    """
    Main function to test the EvaluationAgent.
    """
    # Check if the API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please create a .env file in the tests directory with your OpenAI API key.")
        print("Example: OPENAI_API_KEY=your-api-key-here")
        return

    # Create an EvaluationAgent
    agent = EvaluationAgent(
        system_prompt="You are a critical evaluator specialized in assessing the quality of technical explanations."
    )

    # Define a prompt and a response to evaluate
    prompt = "Explain how neural networks work in simple terms."
    
    good_response = """
    Neural networks are computing systems inspired by the human brain. They consist of interconnected nodes (neurons) 
    organized in layers. Each connection has a weight that adjusts as the network learns.
    
    The process works like this:
    1. Input data enters through the input layer
    2. It's processed through hidden layers where each neuron applies a mathematical function
    3. The output layer produces the final result
    
    Neural networks learn by adjusting connection weights based on the difference between predicted and actual outputs,
    a process called backpropagation. This allows them to recognize patterns and make predictions without explicit programming.
    """
    
    poor_response = "Neural networks are like brains and they learn stuff."
    
    print("Testing EvaluationAgent")
    print("-" * 50)
    
    try:
        # Define custom evaluation criteria
        criteria = ["Accuracy", "Completeness", "Clarity", "Simplicity"]
        
        # Evaluate the good response
        print(f"Evaluating good response against criteria: {', '.join(criteria)}")
        good_evaluation = agent.evaluate(prompt, good_response, criteria)
        
        # Print the evaluation
        print("Evaluation of good response:")
        print(good_evaluation)
        print("-" * 50)
        
        # Evaluate the poor response
        print(f"Evaluating poor response against criteria: {', '.join(criteria)}")
        poor_evaluation = agent.evaluate(prompt, poor_response, criteria)
        
        # Print the evaluation
        print("Evaluation of poor response:")
        print(poor_evaluation)
        print("-" * 50)
        
        # Evaluate with default criteria
        print("Evaluating with default criteria")
        default_evaluation = agent.evaluate(prompt, good_response)
        
        # Print the evaluation
        print("Evaluation with default criteria:")
        print(default_evaluation)
        print("-" * 50)
        
        # Verify the evaluations are not empty
        assert good_evaluation and poor_evaluation and default_evaluation, "One of the evaluations is empty"
        
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()