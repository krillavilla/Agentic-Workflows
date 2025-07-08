"""
Test script for RoutingAgent.

This script demonstrates how to use the RoutingAgent and verifies its functionality.
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

# Import the RoutingAgent
from workflow_agents.base_agents import RoutingAgent, DirectPromptAgent

def product_handler(prompt):
    """Handler for product-related queries."""
    return f"PRODUCT TEAM: We'll look into your request about: {prompt}"

def technical_handler(prompt):
    """Handler for technical queries."""
    return f"TECHNICAL TEAM: We'll address your technical question about: {prompt}"

def support_handler(prompt):
    """Handler for support queries."""
    return f"SUPPORT TEAM: We'll help you with your support issue: {prompt}"

def main():
    """
    Main function to test the RoutingAgent.
    """
    # Check if the API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please create a .env file in the tests directory with your OpenAI API key.")
        print("Example: OPENAI_API_KEY=your-api-key-here")
        return

    # Create a RoutingAgent
    agent = RoutingAgent(
        system_prompt="You are a customer service routing assistant. Your job is to analyze customer requests and route them to the appropriate department."
    )

    # Add routes to the agent with func key
    agent.add_route(
        name="product",
        description="Handles questions about product features, pricing, and availability.",
        handler=product_handler
    )

    agent.add_route(
        name="technical",
        description="Handles technical issues, bugs, and implementation questions.",
        handler=technical_handler
    )

    agent.add_route(
        name="support",
        description="Handles account issues, billing problems, and general customer support.",
        handler=support_handler
    )

    # Set up agents list with func key for embedding-based routing
    agent.agents = [
        {
            "name": "product",
            "description": "Handles questions about product features, pricing, and availability.",
            "expertise": ["product features", "pricing", "availability", "plans", "subscriptions"],
            "func": product_handler
        },
        {
            "name": "technical",
            "description": "Handles technical issues, bugs, and implementation questions.",
            "expertise": ["technical issues", "bugs", "errors", "implementation", "API", "integration"],
            "func": technical_handler
        },
        {
            "name": "support",
            "description": "Handles account issues, billing problems, and general customer support.",
            "expertise": ["account issues", "billing", "password reset", "customer support", "help"],
            "func": support_handler
        }
    ]

    # Define test prompts
    product_prompt = "I'm interested in learning more about your premium plan features."
    technical_prompt = "I'm getting an error when I try to integrate your API with my application."
    support_prompt = "I need help resetting my account password."
    ambiguous_prompt = "I have a question about your service."

    print("Testing RoutingAgent")
    print("-" * 50)

    try:
        # Create output directory
        output_dir = os.path.join(os.path.dirname(__file__), "outputs")
        os.makedirs(output_dir, exist_ok=True)

        # Test routing for product query
        print(f"Testing routing for product query: '{product_prompt}'")
        product_route = agent.respond(product_prompt)
        print("Routing decision:")
        print(product_route)
        print("-" * 50)

        # Generate output file for product routing
        output_content1 = format_output_file(
            prompt=f"Route this query: '{product_prompt}'",
            response=product_route,
            agent_type="RoutingAgent"
        )

        # Save output file
        output_file1 = os.path.join(output_dir, "routing_agent_output_product.txt")
        save_output_file(output_content1, output_file1)
        print(f"Output saved to {output_file1}")

        # Test routing for technical query
        print(f"Testing routing for technical query: '{technical_prompt}'")
        technical_route = agent.respond(technical_prompt)
        print("Routing decision:")
        print(technical_route)
        print("-" * 50)

        # Generate output file for technical routing
        output_content2 = format_output_file(
            prompt=f"Route this query: '{technical_prompt}'",
            response=technical_route,
            agent_type="RoutingAgent"
        )

        # Save output file
        output_file2 = os.path.join(output_dir, "routing_agent_output_technical.txt")
        save_output_file(output_content2, output_file2)
        print(f"Output saved to {output_file2}")

        # Test routing for support query
        print(f"Testing routing for support query: '{support_prompt}'")
        support_route = agent.respond(support_prompt)
        print("Routing decision:")
        print(support_route)
        print("-" * 50)

        # Generate output file for support routing
        output_content3 = format_output_file(
            prompt=f"Route this query: '{support_prompt}'",
            response=support_route,
            agent_type="RoutingAgent"
        )

        # Save output file
        output_file3 = os.path.join(output_dir, "routing_agent_output_support.txt")
        save_output_file(output_content3, output_file3)
        print(f"Output saved to {output_file3}")

        # Test routing for ambiguous query
        print(f"Testing routing for ambiguous query: '{ambiguous_prompt}'")
        ambiguous_route = agent.respond(ambiguous_prompt)
        print("Routing decision:")
        print(ambiguous_route)
        print("-" * 50)

        # Generate output file for ambiguous routing
        output_content4 = format_output_file(
            prompt=f"Route this query: '{ambiguous_prompt}'",
            response=ambiguous_route,
            agent_type="RoutingAgent"
        )

        # Save output file
        output_file4 = os.path.join(output_dir, "routing_agent_output_ambiguous.txt")
        save_output_file(output_content4, output_file4)
        print(f"Output saved to {output_file4}")

        # Verify the routing decisions are not empty
        assert product_route and technical_route and support_route and ambiguous_route, "One of the routing decisions is empty"

        # Test the process method with a direct handler
        print("Testing process method with a direct handler")
        # For testing purposes, we'll use a DirectPromptAgent as a fallback handler
        fallback_agent = DirectPromptAgent(system_prompt="You are a general customer service assistant.")
        agent.add_route(
            name="default",
            description="Handles general inquiries that don't fit other categories.",
            handler=lambda prompt: fallback_agent.respond(prompt)
        )

        # Add default to agents list with func key
        agent.agents.append({
            "name": "default",
            "description": "Handles general inquiries that don't fit other categories.",
            "expertise": ["general", "information", "question", "inquiry"],
            "func": lambda prompt: fallback_agent.respond(prompt)
        })

        # Test the process method
        process_prompt = "Can you tell me more about your company?"
        process_result = agent.process(process_prompt)
        print(f"Process result for '{process_prompt}':")
        print(process_result)
        print("-" * 50)

        # Generate output file for process result
        output_content5 = format_output_file(
            prompt=f"Process this query: '{process_prompt}'",
            response=process_result,
            agent_type="RoutingAgent"
        )

        # Save output file
        output_file5 = os.path.join(output_dir, "routing_agent_output_process.txt")
        save_output_file(output_content5, output_file5)
        print(f"Output saved to {output_file5}")

        print("Test completed successfully!")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
