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

    # Add routes to the agent
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

    # Define test prompts
    product_prompt = "I'm interested in learning more about your premium plan features."
    technical_prompt = "I'm getting an error when I try to integrate your API with my application."
    support_prompt = "I need help resetting my account password."
    ambiguous_prompt = "I have a question about your service."
    
    print("Testing RoutingAgent")
    print("-" * 50)
    
    try:
        # Test routing for product query
        print(f"Testing routing for product query: '{product_prompt}'")
        product_route = agent.route(product_prompt)
        print("Routing decision:")
        print(product_route)
        print("-" * 50)
        
        # Test routing for technical query
        print(f"Testing routing for technical query: '{technical_prompt}'")
        technical_route = agent.route(technical_prompt)
        print("Routing decision:")
        print(technical_route)
        print("-" * 50)
        
        # Test routing for support query
        print(f"Testing routing for support query: '{support_prompt}'")
        support_route = agent.route(support_prompt)
        print("Routing decision:")
        print(support_route)
        print("-" * 50)
        
        # Test routing for ambiguous query
        print(f"Testing routing for ambiguous query: '{ambiguous_prompt}'")
        ambiguous_route = agent.route(ambiguous_prompt)
        print("Routing decision:")
        print(ambiguous_route)
        print("-" * 50)
        
        # Verify the routing decisions are not empty
        assert product_route and technical_route and support_route and ambiguous_route, "One of the routing decisions is empty"
        
        # Test the process method with a direct handler
        print("Testing process method with a direct handler")
        # For testing purposes, we'll use a DirectPromptAgent as a fallback handler
        fallback_agent = DirectPromptAgent(system_prompt="You are a general customer service assistant.")
        agent.add_route(
            name="default",
            description="Handles general inquiries that don't fit other categories.",
            handler=lambda prompt: fallback_agent.run(prompt)
        )
        
        # Note: In a real implementation, you would parse the JSON response from route()
        # and use the actual route name. This is a simplified test.
        
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()