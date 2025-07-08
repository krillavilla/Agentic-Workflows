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
    # Define the agents as a list of dictionaries
    agents = [
        {
            "name": "product_manager",
            "description": "Handles product strategy, requirements, and stakeholder coordination.",
            "expertise": ["product vision", "feature prioritization", "market analysis", "customer needs", 
                         "product roadmap", "competitive analysis", "market positioning"]
        },
        {
            "name": "program_manager",
            "description": "Handles project planning, scheduling, and cross-team coordination.",
            "expertise": ["project timelines", "resource allocation", "team coordination", 
                         "project dependencies", "status updates", "risk management", 
                         "cross-team collaboration"]
        },
        {
            "name": "development_engineer",
            "description": "Handles technical implementation, architecture, and problem-solving.",
            "expertise": ["technical implementation", "code issues", "system architecture", 
                         "technical feasibility", "performance optimization", "API specifications"]
        }
    ]

    # Create the routing agent with the system prompt augmented by the routing knowledge
    agent = RoutingAgent(
        system_prompt=f"""You are an email routing assistant. Your job is to analyze product specification emails 
        and route them to the appropriate team member based on the following knowledge:

        {routing_knowledge}
        """
    )

    # Set the agents attribute
    agent.agents = agents

    return agent

def create_evaluation_agent():
    """
    Create a general evaluation agent.

    Returns:
        EvaluationAgent: The evaluation agent
    """
    return EvaluationAgent(
        system_prompt="You are a quality assurance evaluator. Your job is to assess the quality and appropriateness of responses to product specification emails."
    )

def create_product_manager_evaluation_agent():
    """
    Create a specialized evaluation agent for product manager responses.

    Returns:
        EvaluationAgent: The specialized evaluation agent
    """
    persona = """
    As a Product Management Evaluator, you have:
    - Deep understanding of product strategy and market positioning
    - Experience in feature prioritization and roadmap planning
    - Knowledge of user experience principles and customer needs analysis
    - Ability to assess business value and market impact
    - Expertise in evaluating product requirements clarity and completeness
    """

    return EvaluationAgent(
        system_prompt=f"""You are a specialized evaluator for Product Manager responses.

        {persona}

        Your job is to assess how well the Product Manager's response addresses strategic product concerns,
        demonstrates customer focus, shows market awareness, provides clarity, and offers actionable next steps.
        """
    )

def create_program_manager_evaluation_agent():
    """
    Create a specialized evaluation agent for program manager responses.

    Returns:
        EvaluationAgent: The specialized evaluation agent
    """
    persona = """
    As a Program Management Evaluator, you have:
    - Expertise in project planning and timeline assessment
    - Experience in resource allocation and team coordination
    - Knowledge of risk management and mitigation strategies
    - Ability to evaluate communication effectiveness across teams
    - Understanding of project dependencies and critical path analysis
    """

    return EvaluationAgent(
        system_prompt=f"""You are a specialized evaluator for Program Manager responses.

        {persona}

        Your job is to assess how well the Program Manager's response addresses project planning,
        resource management, risk assessment, timeline accuracy, and provides clear next steps.
        """
    )

def create_development_engineer_evaluation_agent():
    """
    Create a specialized evaluation agent for development engineer responses.

    Returns:
        EvaluationAgent: The specialized evaluation agent
    """
    persona = """
    As a Development Engineering Evaluator, you have:
    - Deep technical knowledge across multiple programming languages and frameworks
    - Experience in system architecture and design patterns
    - Expertise in code quality assessment and performance optimization
    - Understanding of security best practices and vulnerability assessment
    - Knowledge of scalability and maintainability principles
    """

    return EvaluationAgent(
        system_prompt=f"""You are a specialized evaluator for Development Engineer responses.

        {persona}

        Your job is to assess how well the Development Engineer's response addresses technical accuracy,
        implementation feasibility, code quality considerations, performance implications, and security awareness.
        """
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

def product_manager_support_function(email_content):
    """
    Support function for product manager tasks.

    Args:
        email_content (str): The content of the email to process

    Returns:
        dict: The processing results
    """
    # Create the product manager agent
    product_manager = create_product_manager_agent()

    # Generate response
    response = product_manager.run(email_content)

    # Create specialized evaluation agent for product manager
    evaluation_agent = create_product_manager_evaluation_agent()

    # Evaluate the response
    evaluation_result = evaluation_agent.evaluate(
        email_content,
        response,
        ["Strategic Thinking", "Customer Focus", "Market Awareness", "Clarity", "Actionability"]
    )

    # Generate user stories
    user_stories = generate_user_stories(email_content, response)

    return {
        "response": response,
        "evaluation": evaluation_result,
        "user_stories": user_stories
    }

def program_manager_support_function(email_content):
    """
    Support function for program manager tasks.

    Args:
        email_content (str): The content of the email to process

    Returns:
        dict: The processing results
    """
    # Create the program manager agent
    program_manager = create_program_manager_agent()

    # Generate response
    response = program_manager.run(email_content)

    # Create specialized evaluation agent for program manager
    evaluation_agent = create_program_manager_evaluation_agent()

    # Evaluate the response
    evaluation_result = evaluation_agent.evaluate(
        email_content,
        response,
        ["Project Planning", "Resource Management", "Risk Assessment", "Timeline Accuracy", "Clarity"]
    )

    # Generate features
    features = generate_features(email_content, response)

    return {
        "response": response,
        "evaluation": evaluation_result,
        "features": features
    }

def development_engineer_support_function(email_content):
    """
    Support function for development engineer tasks.

    Args:
        email_content (str): The content of the email to process

    Returns:
        dict: The processing results
    """
    # Create the development engineer agent
    development_engineer = create_development_engineer_agent()

    # Generate response
    response = development_engineer.run(email_content)

    # Create specialized evaluation agent for development engineer
    evaluation_agent = create_development_engineer_evaluation_agent()

    # Evaluate the response
    evaluation_result = evaluation_agent.evaluate(
        email_content,
        response,
        ["Technical Accuracy", "Implementation Feasibility", "Code Quality", "Performance Consideration", "Security Awareness"]
    )

    # Generate engineering tasks
    engineering_tasks = generate_engineering_tasks(email_content, response)

    return {
        "response": response,
        "evaluation": evaluation_result,
        "engineering_tasks": engineering_tasks
    }

def generate_user_stories(email_content, response):
    """
    Generate user stories based on email content and response.

    Args:
        email_content (str): The content of the email
        response (str): The response from the product manager

    Returns:
        list: List of user stories
    """
    # Create a specialized agent for generating user stories
    agent = KnowledgeAugmentedPromptAgent(
        system_prompt="You are a product manager specialized in creating user stories.",
        knowledge="""
        User stories should follow the format:
        As a [type of user], I want [an action] so that [a benefit/value].

        Good user stories are:
        - Independent
        - Negotiable
        - Valuable
        - Estimable
        - Small
        - Testable
        """
    )

    prompt = f"""
    Based on the following email and response, generate 3-5 user stories that capture the requirements:

    EMAIL:
    {email_content}

    RESPONSE:
    {response}

    Format each user story as:
    - As a [type of user], I want [an action] so that [a benefit/value].

    Return the user stories as a JSON array of strings.
    """

    result = agent.run(prompt)

    # Try to parse the result as JSON, if it fails, return it as a string
    try:
        return json.loads(result)
    except:
        return [result]

def generate_features(email_content, response):
    """
    Generate features based on email content and response.

    Args:
        email_content (str): The content of the email
        response (str): The response from the program manager

    Returns:
        list: List of features
    """
    # Create a specialized agent for generating features
    agent = KnowledgeAugmentedPromptAgent(
        system_prompt="You are a program manager specialized in defining product features.",
        knowledge="""
        Features should be:
        - Specific and well-defined
        - Measurable
        - Aligned with product goals
        - Realistic to implement
        - Time-bound

        Each feature should include:
        - Name
        - Description
        - Priority (High, Medium, Low)
        - Estimated effort
        """
    )

    prompt = f"""
    Based on the following email and response, generate 3-5 features that should be implemented:

    EMAIL:
    {email_content}

    RESPONSE:
    {response}

    Format each feature as a JSON object with the following properties:
    - name: The name of the feature
    - description: A brief description of the feature
    - priority: The priority (High, Medium, Low)
    - effort: Estimated effort (Small, Medium, Large)

    Return the features as a JSON array of objects.
    """

    result = agent.run(prompt)

    # Try to parse the result as JSON, if it fails, return it as a string
    try:
        return json.loads(result)
    except:
        return [result]

def generate_engineering_tasks(email_content, response):
    """
    Generate engineering tasks based on email content and response.

    Args:
        email_content (str): The content of the email
        response (str): The response from the development engineer

    Returns:
        list: List of engineering tasks
    """
    # Create a specialized agent for generating engineering tasks
    agent = KnowledgeAugmentedPromptAgent(
        system_prompt="You are a development engineer specialized in breaking down technical requirements into tasks.",
        knowledge="""
        Engineering tasks should be:
        - Specific and actionable
        - Small enough to be completed in 1-3 days
        - Technical in nature
        - Testable
        - Independent when possible

        Each task should include:
        - Name
        - Description
        - Technical requirements
        - Estimated complexity
        """
    )

    prompt = f"""
    Based on the following email and response, generate 3-5 engineering tasks that need to be completed:

    EMAIL:
    {email_content}

    RESPONSE:
    {response}

    Format each task as a JSON object with the following properties:
    - name: The name of the task
    - description: A brief description of what needs to be done
    - requirements: Technical requirements or considerations
    - complexity: Estimated complexity (Simple, Moderate, Complex)

    Return the tasks as a JSON array of objects.
    """

    result = agent.run(prompt)

    # Try to parse the result as JSON, if it fails, return it as a string
    try:
        return json.loads(result)
    except:
        return [result]

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

    # Initialize a list to accumulate completed steps
    completed_steps = []

    # Load the routing knowledge
    routing_knowledge = load_routing_knowledge()
    print("Loaded routing knowledge")

    # Create the agents
    product_manager = create_product_manager_agent()
    program_manager = create_program_manager_agent()
    development_engineer = create_development_engineer_agent()

    # Create the routing agent with knowledge from the router file and configure with agents list
    routing_agent = create_routing_agent(routing_knowledge)

    # Create specialized evaluation agents
    product_manager_eval = create_product_manager_evaluation_agent()
    program_manager_eval = create_program_manager_evaluation_agent()
    development_engineer_eval = create_development_engineer_evaluation_agent()

    # Create the action planning agent
    action_planning_agent = create_action_planning_agent()

    # Add actions to the action planning agent
    action_planning_agent.add_action(
        name="route_email",
        description="Route the email to the appropriate team member. Parameters: email_content (str)",
        handler=lambda email_content: routing_agent.route(email_content)
    )

    action_planning_agent.add_action(
        name="process_with_product_manager",
        description="Process the email with the product manager. Parameters: email_content (str)",
        handler=lambda email_content: product_manager_support_function(email_content)
    )

    action_planning_agent.add_action(
        name="process_with_program_manager",
        description="Process the email with the program manager. Parameters: email_content (str)",
        handler=lambda email_content: program_manager_support_function(email_content)
    )

    action_planning_agent.add_action(
        name="process_with_development_engineer",
        description="Process the email with the development engineer. Parameters: email_content (str)",
        handler=lambda email_content: development_engineer_support_function(email_content)
    )

    action_planning_agent.add_action(
        name="generate_consolidated_output",
        description="Generate a consolidated output from all the processing results. Parameters: results (list)",
        handler=lambda results: generate_consolidated_output(results)
    )

    # Generate a plan for processing the email
    print("Generating workflow plan...")
    plan_response = action_planning_agent.plan(f"Process the following email and generate a comprehensive response with appropriate routing: {email_content}")
    print(f"Plan generated: {plan_response}")
    print("-" * 50)

    # Parse the plan
    try:
        # Try to parse the plan as JSON
        plan = json.loads(plan_response)
    except:
        # If parsing fails, create a default plan
        print("Failed to parse plan. Using default workflow.")
        plan = [
            {
                "action": "route_email",
                "parameters": {"email_content": email_content},
                "explanation": "First, we need to determine which team member should handle this email."
            },
            {
                "action": "process_with_product_manager",
                "parameters": {"email_content": email_content},
                "explanation": "Process with product manager as default."
            },
            {
                "action": "generate_consolidated_output",
                "parameters": {"results": []},
                "explanation": "Generate the final output."
            }
        ]

    # Execute the plan
    results = []
    route = None

    for step in plan:
        action_name = step.get("action")
        parameters = step.get("parameters", {})
        explanation = step.get("explanation", "")

        print(f"Executing step: {action_name} - {explanation}")

        try:
            if action_name == "route_email":
                # Execute routing
                routing_result = routing_agent.route(email_content)
                print(f"Routing result: {routing_result}")

                # Try to parse the routing result to get the route
                try:
                    routing_json = json.loads(routing_result)
                    route = routing_json.get("route", "product_manager")
                except:
                    # If parsing fails, use default route
                    print("Failed to parse routing result. Using default route.")
                    route = "product_manager"

                # Add to completed steps
                completed_steps.append({
                    "step": "routing",
                    "result": routing_result,
                    "route": route
                })

            elif action_name == "process_with_product_manager":
                # Process with product manager
                pm_result = product_manager_support_function(email_content)
                results.append({"role": "product_manager", "result": pm_result})

                # Add to completed steps
                completed_steps.append({
                    "step": "product_manager_processing",
                    "response": pm_result.get("response"),
                    "evaluation": pm_result.get("evaluation"),
                    "user_stories": pm_result.get("user_stories")
                })

            elif action_name == "process_with_program_manager":
                # Process with program manager
                pgm_result = program_manager_support_function(email_content)
                results.append({"role": "program_manager", "result": pgm_result})

                # Add to completed steps
                completed_steps.append({
                    "step": "program_manager_processing",
                    "response": pgm_result.get("response"),
                    "evaluation": pgm_result.get("evaluation"),
                    "features": pgm_result.get("features")
                })

            elif action_name == "process_with_development_engineer":
                # Process with development engineer
                de_result = development_engineer_support_function(email_content)
                results.append({"role": "development_engineer", "result": de_result})

                # Add to completed steps
                completed_steps.append({
                    "step": "development_engineer_processing",
                    "response": de_result.get("response"),
                    "evaluation": de_result.get("evaluation"),
                    "engineering_tasks": de_result.get("engineering_tasks")
                })

            elif action_name == "generate_consolidated_output":
                # Generate consolidated output
                consolidated_output = generate_consolidated_output(results)

                # Add to completed steps
                completed_steps.append({
                    "step": "consolidated_output",
                    "output": consolidated_output
                })
        except Exception as e:
            print(f"Error executing step {action_name}: {str(e)}")
            # Add error to completed steps
            completed_steps.append({
                "step": action_name,
                "error": str(e)
            })

    print("-" * 50)
    print("Email processing completed.")

    # Return the results
    return {
        "email": email_content,
        "completed_steps": completed_steps,
        "final_output": completed_steps[-1] if completed_steps else None
    }

def generate_consolidated_output(results):
    """
    Generate a consolidated, structured project plan output from the processing results.

    Args:
        results (list): List of processing results from different agents

    Returns:
        dict: Consolidated output with user stories, features, and engineering tasks
    """
    # Create a specialized agent for consolidating the output
    agent = KnowledgeAugmentedPromptAgent(
        system_prompt="You are a project planning specialist responsible for creating comprehensive project plans.",
        knowledge="""
        A good project plan includes:
        1. User stories that capture the requirements from the user's perspective
        2. Features that define what will be built to satisfy the user stories
        3. Engineering tasks that break down the technical work needed to implement the features

        The plan should be well-organized, prioritized, and aligned across all three levels.
        """
    )

    # Extract user stories, features, and engineering tasks from the results
    user_stories = []
    features = []
    engineering_tasks = []

    for result_item in results:
        role = result_item.get("role")
        result = result_item.get("result", {})

        if role == "product_manager" and "user_stories" in result:
            user_stories.extend(result["user_stories"])
        elif role == "program_manager" and "features" in result:
            features.extend(result["features"])
        elif role == "development_engineer" and "engineering_tasks" in result:
            engineering_tasks.extend(result["engineering_tasks"])

    # Create a consolidated output
    consolidated_output = {
        "user_stories": user_stories,
        "features": features,
        "engineering_tasks": engineering_tasks
    }

    # If any section is empty, generate placeholder content
    if not user_stories:
        consolidated_output["user_stories"] = ["No user stories were generated."]

    if not features:
        consolidated_output["features"] = ["No features were defined."]

    if not engineering_tasks:
        consolidated_output["engineering_tasks"] = ["No engineering tasks were created."]

    # Add a summary of the project plan
    prompt = f"""
    Create a brief summary of the following project plan:

    USER STORIES:
    {json.dumps(user_stories, indent=2)}

    FEATURES:
    {json.dumps(features, indent=2)}

    ENGINEERING TASKS:
    {json.dumps(engineering_tasks, indent=2)}

    Your summary should highlight the key aspects of the plan and how the components align.
    """

    summary = agent.run(prompt)
    consolidated_output["summary"] = summary

    return consolidated_output

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

    # Process each email and display the results
    print("\nProcessing Product Manager Email:")
    product_result = process_email(product_email)
    print("\nFinal Output:")
    if product_result.get("final_output"):
        print(json.dumps(product_result["final_output"], indent=2))

    print("\nProcessing Program Manager Email:")
    program_result = process_email(program_email)
    print("\nFinal Output:")
    if program_result.get("final_output"):
        print(json.dumps(program_result["final_output"], indent=2))

    print("\nProcessing Development Engineer Email:")
    technical_result = process_email(technical_email)
    print("\nFinal Output:")
    if technical_result.get("final_output"):
        print(json.dumps(technical_result["final_output"], indent=2))

    # Generate a comprehensive project plan from all emails
    print("\nGenerating Comprehensive Project Plan:")

    # Combine all completed steps from all emails
    all_completed_steps = []
    all_completed_steps.extend(product_result.get("completed_steps", []))
    all_completed_steps.extend(program_result.get("completed_steps", []))
    all_completed_steps.extend(technical_result.get("completed_steps", []))

    # Extract all results for consolidated output
    all_results = []
    for step in all_completed_steps:
        if step.get("step") in ["product_manager_processing", "program_manager_processing", "development_engineer_processing"]:
            role = step.get("step").replace("_processing", "")
            result = {
                "response": step.get("response"),
                "evaluation": step.get("evaluation")
            }

            if "user_stories" in step:
                result["user_stories"] = step.get("user_stories")
            elif "features" in step:
                result["features"] = step.get("features")
            elif "engineering_tasks" in step:
                result["engineering_tasks"] = step.get("engineering_tasks")

            all_results.append({"role": role, "result": result})

    # Generate consolidated output
    comprehensive_plan = generate_consolidated_output(all_results)

    print("\nComprehensive Project Plan:")
    print(json.dumps(comprehensive_plan, indent=2))

if __name__ == "__main__":
    main()
