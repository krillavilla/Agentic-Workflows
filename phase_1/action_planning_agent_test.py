"""
Test script for ActionPlanningAgent.

This script demonstrates how to use the ActionPlanningAgent and verifies its functionality.
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

# Import the ActionPlanningAgent
from workflow_agents.base_agents import ActionPlanningAgent

# Define some example actions for the agent to use

def search_database(query, limit=10):
    """
    Simulates searching a database.

    Args:
        query (str): The search query
        limit (int): Maximum number of results to return

    Returns:
        str: Search results
    """
    print(f"Searching database for '{query}' with limit {limit}")
    return f"Found {min(limit, 5)} results for '{query}'"

def send_email(to, subject, body):
    """
    Simulates sending an email.

    Args:
        to (str): Recipient email
        subject (str): Email subject
        body (str): Email body

    Returns:
        str: Confirmation message
    """
    print(f"Sending email to {to}")
    print(f"Subject: {subject}")
    print(f"Body: {body}")
    return f"Email sent to {to}"

def generate_report(data, format="pdf"):
    """
    Simulates generating a report.

    Args:
        data (str): Data to include in the report
        format (str): Report format (pdf, html, txt)

    Returns:
        str: Report generation confirmation
    """
    print(f"Generating {format} report with data: {data}")
    return f"Report generated in {format} format"

def main():
    """
    Main function to test the ActionPlanningAgent.
    """
    # Check if the API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please create a .env file in the tests directory with your OpenAI API key.")
        print("Example: OPENAI_API_KEY=your-api-key-here")
        return

    # Create an ActionPlanningAgent
    agent = ActionPlanningAgent(
        system_prompt="You are an action planning assistant specialized in helping users accomplish tasks efficiently."
    )

    # Add actions to the agent
    agent.add_action(
        name="search_database",
        description="Search a database for information. Parameters: query (str), limit (int, optional)",
        handler=search_database
    )

    agent.add_action(
        name="send_email",
        description="Send an email. Parameters: to (str), subject (str), body (str)",
        handler=send_email
    )

    agent.add_action(
        name="generate_report",
        description="Generate a report. Parameters: data (str), format (str, optional)",
        handler=generate_report
    )

    # Define a task for the agent
    task = "Find information about renewable energy, send an email to team@example.com with the findings, and generate a PDF report."

    print(f"Testing ActionPlanningAgent with task: '{task}'")
    print("-" * 50)

    try:
        # Create output directory
        output_dir = os.path.join(os.path.dirname(__file__), "outputs")
        os.makedirs(output_dir, exist_ok=True)

        # Use respond method to generate a plan for the task
        plan_response = agent.respond(task)

        print("Generated Plan:")
        print(plan_response)
        print("-" * 50)

        # Verify the plan is not empty
        assert plan_response, "Plan is empty"

        # Generate output file for first task
        output_content1 = format_output_file(
            prompt=f"Create a plan for this task: '{task}'",
            response=plan_response,
            agent_type="ActionPlanningAgent"
        )

        # Save output file
        output_file1 = os.path.join(output_dir, "action_planning_agent_output_task1.txt")
        save_output_file(output_content1, output_file1)
        print(f"Output saved to {output_file1}")

        # Test with a different task
        task2 = "Search for the latest AI research papers and email a summary to research@example.com."

        print(f"Testing with different task: '{task2}'")
        plan_response2 = agent.respond(task2)

        print("Generated Plan for second task:")
        print(plan_response2)
        print("-" * 50)

        # Verify the second plan is not empty
        assert plan_response2, "Second plan is empty"

        # Generate output file for second task
        output_content2 = format_output_file(
            prompt=f"Create a plan for this task: '{task2}'",
            response=plan_response2,
            agent_type="ActionPlanningAgent"
        )

        # Save output file
        output_file2 = os.path.join(output_dir, "action_planning_agent_output_task2.txt")
        save_output_file(output_content2, output_file2)
        print(f"Output saved to {output_file2}")

        # Execute the plan for the first task
        print("Executing plan for first task:")
        execution_result = agent.run(task)

        print("Execution result:")
        print(json.dumps(execution_result, indent=2))
        print("-" * 50)

        # Generate output file for execution result
        output_content3 = format_output_file(
            prompt=f"Execute this task: '{task}'",
            response=json.dumps(execution_result, indent=2),
            agent_type="ActionPlanningAgent"
        )

        # Save output file
        output_file3 = os.path.join(output_dir, "action_planning_agent_output_execution.txt")
        save_output_file(output_content3, output_file3)
        print(f"Output saved to {output_file3}")

        print("Test completed successfully!")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
