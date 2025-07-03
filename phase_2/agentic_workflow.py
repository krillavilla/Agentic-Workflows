"""
Agentic Workflow for Product Specification Email Routing

This module implements an agentic workflow that routes product specification
emails to the appropriate team member and generates responses.
"""

import os
import sys
import json
from dotenv import load_dotenv

# Add the parent directory to the path so we can import the workflow_agents package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
load_dotenv()

# Import the agent classes
from phase_1.workflow_agents.base_agents import (
    DirectPromptAgent,
    AugmentedPromptAgent,
    KnowledgeAugmentedPromptAgent,
    RAGKnowledgePromptAgent,
    EvaluationAgent,
    RoutingAgent,
    ActionPlanningAgent
)

def load_routing_knowledge():
    """
    Load the routing knowledge from the Product-Spec-Email-Router.txt file.
    
    Returns:
        str: The content of the routing file
    """
    router_file_path = os.path.join(os.path.dirname(__file__), "Product-Spec-Email-Router.txt")
    with open(router_file_path, 'r') as file:
        return file.read()

def create_product_manager_agent():
    """
    Create an agent for handling product manager tasks.
    
    Returns:
        KnowledgeAugmentedPromptAgent: The product manager agent
    """
    product_knowledge = """
    As a Product Manager, you are responsible for:
    - Defining product vision and strategy
    - Prioritizing features based on customer needs and business goals
    - Creating and maintaining product requirements
    - Coordinating with stakeholders to ensure alignment
    - Analyzing market trends and customer feedback
    
    When responding to inquiries:
    - Focus on strategic aspects of the product
    - Consider market positioning and competitive landscape
    - Emphasize customer value and business impact
    - Provide clear direction on product priorities
    - Be data-driven in your decision making
    """
    
    return KnowledgeAugmentedPromptAgent(
        system_prompt="You are a Product Manager responsible for product strategy, requirements, and stakeholder coordination.",
        knowledge=product_knowledge
    )

def create_program_manager_agent():
    """
    Create an agent for handling program manager tasks.
    
    Returns:
        KnowledgeAugmentedPromptAgent: The program manager agent
    """
    program_knowledge = """
    As a Program Manager, you are responsible for:
    - Planning and scheduling project activities
    - Allocating resources effectively
    - Coordinating across teams to ensure smooth execution
    - Tracking progress against milestones
    - Identifying and mitigating project risks
    
    When responding to inquiries:
    - Focus on operational aspects of the project
    - Provide clear timelines and dependencies
    - Emphasize coordination and communication
    - Be specific about resource requirements
    - Address risks and mitigation strategies
    """
    
    return KnowledgeAugmentedPromptAgent(
        system_prompt="You are a Program Manager responsible for project planning, scheduling, and cross-team coordination.",
        knowledge=program_knowledge
    )

def create_development_engineer_agent():
    """
    Create an agent for handling development engineer tasks.
    
    Returns:
        KnowledgeAugmentedPromptAgent: The development engineer agent
    """
    engineer_knowledge = """
    As a Development Engineer, you are responsible for:
    - Implementing technical solutions
    - Developing and maintaining code
    - Designing system architecture
    - Evaluating technical feasibility
    - Resolving technical issues and bugs
    
    When responding to inquiries:
    - Focus on technical implementation details
    - Provide specific code or architecture recommendations
    - Consider performance, scalability, and maintainability
    - Be clear about technical constraints and trade-offs
    - Suggest practical solutions to technical challenges
    """
    
    return KnowledgeAugmentedPromptAgent(
        system_prompt="You are a Development Engineer responsible for technical implementation, architecture, and problem-solving.",
        knowledge=engineer_knowledge
    )

def create_routing_agent(routing_knowledge):
    """
    Create an agent for routing emails to the appropriate team member.
    
    Args:
        routing_knowledge (str): The routing knowledge from Product-Spec-Email-Router.txt
        
    Returns:
        RoutingAgent: The routing agent
    """
    agent = RoutingAgent(
        system_prompt="You are an email routing assistant. Your job is to analyze product specification emails and route them to the appropriate team member."
    )
    
    return agent

def create_evaluation_agent():
    """
    Create an agent for evaluating responses.
    
    Returns:
        EvaluationAgent: The evaluation agent
    """
    return EvaluationAgent(
        system_prompt="You are a quality assurance evaluator. Your job is to assess the quality and appropriateness of responses to product specification emails."
    )

def create_action_planning_agent():
    """
    Create an agent for planning and executing the workflow.
    
    Returns:
        ActionPlanningAgent: The action planning agent
    """
    return ActionPlanningAgent(
        system_prompt="You are a workflow coordinator. Your job is to plan and execute the steps needed to process product specification emails."
    )

def process_email(email_content):
    """
    Process an email using the agentic workflow.
    
    Args:
        email_content (str): The content of the email to process
        
    Returns:
        dict: The processing results including routing decision and response
    """
    print("Starting email processing workflow...")
    print("-" * 50)
    
    # Load the routing knowledge
    routing_knowledge = load_routing_knowledge()
    print("Loaded routing knowledge")
    
    # Create the agents
    product_manager = create_product_manager_agent()
    program_manager = create_program_manager_agent()
    development_engineer = create_development_engineer_agent()
    
    # Create the routing agent with knowledge from the router file
    routing_agent = create_routing_agent(routing_knowledge)
    
    # Add routes to the routing agent
    routing_agent.add_route(
        name="product_manager",
        description="Handles product strategy, requirements, and stakeholder coordination.",
        handler=lambda prompt: product_manager.run(prompt)
    )
    
    routing_agent.add_route(
        name="program_manager",
        description="Handles project planning, scheduling, and cross-team coordination.",
        handler=lambda prompt: program_manager.run(prompt)
    )
    
    routing_agent.add_route(
        name="development_engineer",
        description="Handles technical implementation, architecture, and problem-solving.",
        handler=lambda prompt: development_engineer.run(prompt)
    )
    
    # Create the evaluation agent
    evaluation_agent = create_evaluation_agent()
    
    # Create the action planning agent
    action_planning_agent = create_action_planning_agent()
    
    # Add actions to the action planning agent
    action_planning_agent.add_action(
        name="route_email",
        description="Route the email to the appropriate team member. Parameters: email_content (str)",
        handler=lambda email_content: routing_agent.route(email_content)
    )
    
    action_planning_agent.add_action(
        name="generate_response",
        description="Generate a response to the email. Parameters: email_content (str), route (str)",
        handler=lambda email_content, route: {
            "product_manager": product_manager.run(email_content),
            "program_manager": program_manager.run(email_content),
            "development_engineer": development_engineer.run(email_content)
        }.get(route, "No handler found for this route")
    )
    
    action_planning_agent.add_action(
        name="evaluate_response",
        description="Evaluate the quality of the response. Parameters: email_content (str), response (str)",
        handler=lambda email_content, response: evaluation_agent.evaluate(
            email_content, 
            response, 
            ["Relevance", "Completeness", "Clarity", "Actionability"]
        )
    )
    
    # Step 1: Route the email
    print("Step 1: Routing the email...")
    routing_result = routing_agent.route(email_content)
    print(f"Routing result: {routing_result}")
    print("-" * 50)
    
    # In a real implementation, we would parse the JSON response
    # For this example, we'll extract the route manually
    # This is a placeholder - in a real implementation, you would parse the JSON
    route = "product_manager"  # Default route
    
    # Step 2: Generate a response based on the routing
    print(f"Step 2: Generating response using the {route} agent...")
    if route == "product_manager":
        response = product_manager.run(email_content)
    elif route == "program_manager":
        response = program_manager.run(email_content)
    elif route == "development_engineer":
        response = development_engineer.run(email_content)
    else:
        response = "Unable to determine the appropriate team member for this inquiry."
    
    print(f"Generated response: {response}")
    print("-" * 50)
    
    # Step 3: Evaluate the response
    print("Step 3: Evaluating the response...")
    evaluation_result = evaluation_agent.evaluate(
        email_content, 
        response, 
        ["Relevance", "Completeness", "Clarity", "Actionability"]
    )
    print(f"Evaluation result: {evaluation_result}")
    print("-" * 50)
    
    # Return the results
    return {
        "email": email_content,
        "routing": routing_result,
        "response": response,
        "evaluation": evaluation_result
    }

def main():
    """
    Main function to demonstrate the agentic workflow.
    """
    # Check if the API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set the OPENAI_API_KEY environment variable with your OpenAI API key.")
        print("Example: export OPENAI_API_KEY=your-api-key-here")
        return
    
    # Example emails for testing
    product_email = """
    Subject: Feature Prioritization for Q4
    
    Hi team,
    
    We need to decide which features to prioritize for the Q4 release. Based on customer feedback, 
    I think we should focus on enhancing the reporting dashboard and adding export functionality.
    
    Can we schedule a meeting to discuss this and align on the roadmap?
    
    Thanks,
    Sarah
    """
    
    program_email = """
    Subject: Timeline for Payment Integration
    
    Hello,
    
    I'm trying to create a project schedule for the payment integration feature. 
    Can you provide an estimate of when the development work will be completed?
    
    Also, are there any dependencies or potential blockers we should be aware of?
    
    Best regards,
    Michael
    """
    
    technical_email = """
    Subject: API Authentication Issue
    
    Hi dev team,
    
    We're experiencing intermittent 401 errors with the authentication service. 
    The issue seems to occur during peak usage times. Can you investigate what might be 
    causing this and suggest a solution?
    
    I've attached the error logs for reference.
    
    Thanks,
    David
    """
    
    # Process each email
    print("\nProcessing Product Manager Email:")
    process_email(product_email)
    
    print("\nProcessing Program Manager Email:")
    process_email(program_email)
    
    print("\nProcessing Development Engineer Email:")
    process_email(technical_email)

if __name__ == "__main__":
    main()