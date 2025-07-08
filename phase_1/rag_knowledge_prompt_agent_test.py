"""
Test script for RAGKnowledgePromptAgent.

This script demonstrates how to use the RAGKnowledgePromptAgent and verifies its functionality.
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

# Import the RAGKnowledgePromptAgent
from workflow_agents.base_agents import RAGKnowledgePromptAgent

def main():
    """
    Main function to test the RAGKnowledgePromptAgent.
    """
    # Check if the API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please create a .env file in the tests directory with your OpenAI API key.")
        print("Example: OPENAI_API_KEY=your-api-key-here")
        return

    # Define a knowledge base with multiple topics
    knowledge_base = {
        "Python Programming": """
        Python is a high-level, interpreted programming language known for its readability and simplicity.
        Key features of Python include:
        - Dynamic typing and binding
        - Easy-to-read syntax
        - Object-oriented programming support
        - Extensive standard library
        - Large ecosystem of third-party packages

        Python is widely used in web development, data analysis, artificial intelligence, scientific computing, and automation.
        """,

        "Machine Learning": """
        Machine Learning is a subset of artificial intelligence that focuses on developing systems that can learn from data.
        Common machine learning algorithms include:
        - Linear Regression
        - Logistic Regression
        - Decision Trees
        - Random Forests
        - Support Vector Machines
        - Neural Networks

        Popular Python libraries for machine learning include scikit-learn, TensorFlow, and PyTorch.
        """,

        "Web Development": """
        Web development involves creating websites and web applications.
        It typically includes:
        - Frontend development (HTML, CSS, JavaScript)
        - Backend development (server-side logic)
        - Database management

        Popular Python frameworks for web development include Django, Flask, and FastAPI.
        """,

        "Data Science": """
        Data Science combines domain expertise, programming skills, and knowledge of mathematics and statistics.
        Key components of data science include:
        - Data collection and cleaning
        - Exploratory data analysis
        - Feature engineering
        - Model building and evaluation
        - Data visualization

        Popular Python libraries for data science include pandas, NumPy, and Matplotlib.
        """
    }

    # Create a RAGKnowledgePromptAgent with the knowledge base
    agent = RAGKnowledgePromptAgent(
        system_prompt="You are a helpful assistant specialized in providing accurate information on technical topics.",
        knowledge_base=knowledge_base
    )

    # Test the agent with a prompt related to Python
    prompt1 = "What are the key features of Python programming language?"

    print(f"Testing RAGKnowledgePromptAgent with prompt: '{prompt1}'")
    print("-" * 50)

    try:
        # Use the respond method instead of run
        response1 = agent.respond(prompt1)

        # Print the response
        print("Response to Python query:")
        print(response1)
        print("-" * 50)

        # Verify the response is not empty
        assert response1, "Response is empty"

        # Generate output file
        output_content1 = format_output_file(
            prompt=prompt1,
            response=response1,
            agent_type="RAGKnowledgePromptAgent",
            knowledge_source="Python Programming knowledge base"
        )

        # Save output file
        output_dir = os.path.join(os.path.dirname(__file__), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        output_file1 = os.path.join(output_dir, "rag_knowledge_prompt_agent_output_python.txt")
        save_output_file(output_content1, output_file1)
        print(f"Output saved to {output_file1}")

        # Test with a prompt related to machine learning
        prompt2 = "Tell me about machine learning algorithms."
        print(f"Testing with different prompt: '{prompt2}'")

        # Use the respond method instead of run
        response2 = agent.respond(prompt2)

        # Print the response
        print("Response to machine learning query:")
        print(response2)
        print("-" * 50)

        # Generate output file for second prompt
        output_content2 = format_output_file(
            prompt=prompt2,
            response=response2,
            agent_type="RAGKnowledgePromptAgent",
            knowledge_source="Machine Learning knowledge base"
        )

        # Save output file for second prompt
        output_file2 = os.path.join(output_dir, "rag_knowledge_prompt_agent_output_ml.txt")
        save_output_file(output_content2, output_file2)
        print(f"Output saved to {output_file2}")

        # Test with a prompt that doesn't directly match any knowledge
        prompt3 = "Explain the concept of blockchain."
        print(f"Testing with unrelated prompt: '{prompt3}'")

        # Use the respond method instead of run
        response3 = agent.respond(prompt3)

        # Print the response
        print("Response to blockchain query:")
        print(response3)
        print("-" * 50)

        # Generate output file for third prompt
        output_content3 = format_output_file(
            prompt=prompt3,
            response=response3,
            agent_type="RAGKnowledgePromptAgent",
            knowledge_source="No specific knowledge found for blockchain query"
        )

        # Save output file for third prompt
        output_file3 = os.path.join(output_dir, "rag_knowledge_prompt_agent_output_blockchain.txt")
        save_output_file(output_content3, output_file3)
        print(f"Output saved to {output_file3}")

        # Verify all responses are not empty
        assert response2 and response3, "One of the responses is empty"

        print("Test completed successfully!")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
