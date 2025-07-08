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

# Import the EvaluationAgent
from workflow_agents.base_agents import EvaluationAgent, DirectPromptAgent


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

    # Create a DirectPromptAgent to evaluate
    agent_to_evaluate = DirectPromptAgent()

    # Create an EvaluationAgent with the required agent_to_evaluate parameter
    agent = EvaluationAgent(
        system_prompt="You are a critical evaluator specialized in assessing the quality of technical explanations.",
        agent_to_evaluate=agent_to_evaluate
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

        # Set max iterations for evaluation
        max_iterations = 2

        # Evaluate the good response
        print(f"Evaluating good response against criteria: {', '.join(criteria)}")
        good_evaluation_result = agent.evaluate(prompt, good_response, criteria, max_iterations)

        # Extract components from the evaluation result
        good_final_response = good_evaluation_result.get("final_response", "")
        good_evaluation = good_evaluation_result.get("evaluation", {})
        good_iteration_count = good_evaluation_result.get("iteration_count", 0)

        # Print the evaluation
        print("Evaluation of good response:")
        print(f"Final response: {good_final_response}")
        print(f"Evaluation: {good_evaluation}")
        print(f"Iteration count: {good_iteration_count}")
        print("-" * 50)

        # Generate output file for good response evaluation
        output_content1 = format_output_file(
            prompt=f"Evaluate this response to '{prompt}':\n\n{good_response}",
            response=json.dumps(good_evaluation_result, indent=2),
            agent_type="EvaluationAgent"
        )

        # Save output file
        output_dir = os.path.join(os.path.dirname(__file__), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        output_file1 = os.path.join(output_dir, "evaluation_agent_output_good.txt")
        save_output_file(output_content1, output_file1)
        print(f"Output saved to {output_file1}")

        # Evaluate the poor response
        print(f"Evaluating poor response against criteria: {', '.join(criteria)}")
        poor_evaluation_result = agent.evaluate(prompt, poor_response, criteria, max_iterations)

        # Extract components from the evaluation result
        poor_final_response = poor_evaluation_result.get("final_response", "")
        poor_evaluation = poor_evaluation_result.get("evaluation", {})
        poor_iteration_count = poor_evaluation_result.get("iteration_count", 0)

        # Print the evaluation
        print("Evaluation of poor response:")
        print(f"Final response: {poor_final_response}")
        print(f"Evaluation: {poor_evaluation}")
        print(f"Iteration count: {poor_iteration_count}")
        print("-" * 50)

        # Generate output file for poor response evaluation
        output_content2 = format_output_file(
            prompt=f"Evaluate this response to '{prompt}':\n\n{poor_response}",
            response=json.dumps(poor_evaluation_result, indent=2),
            agent_type="EvaluationAgent"
        )

        # Save output file
        output_file2 = os.path.join(output_dir, "evaluation_agent_output_poor.txt")
        save_output_file(output_content2, output_file2)
        print(f"Output saved to {output_file2}")

        # Evaluate with default criteria
        print("Evaluating with default criteria")
        default_evaluation_result = agent.evaluate(prompt, good_response)

        # Extract components from the evaluation result
        default_final_response = default_evaluation_result.get("final_response", "")
        default_evaluation = default_evaluation_result.get("evaluation", {})
        default_iteration_count = default_evaluation_result.get("iteration_count", 0)

        # Print the evaluation
        print("Evaluation with default criteria:")
        print(f"Final response: {default_final_response}")
        print(f"Evaluation: {default_evaluation}")
        print(f"Iteration count: {default_iteration_count}")
        print("-" * 50)

        # Generate output file for default criteria evaluation
        output_content3 = format_output_file(
            prompt=f"Evaluate this response to '{prompt}' with default criteria:\n\n{good_response}",
            response=json.dumps(default_evaluation_result, indent=2),
            agent_type="EvaluationAgent"
        )

        # Save output file
        output_file3 = os.path.join(output_dir, "evaluation_agent_output_default.txt")
        save_output_file(output_content3, output_file3)
        print(f"Output saved to {output_file3}")

        # Verify the evaluations are not empty
        assert good_evaluation and poor_evaluation and default_evaluation, "One of the evaluations is empty"

        print("Test completed successfully!")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()