"""
Base agent classes for workflow automation.

This module contains various agent implementations that can be used
in agentic workflows. All agents use the OpenAI API with gpt-3.5-turbo model.
"""

import os
import json
from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai import BadRequestError
from typing import List, Dict, Any, Optional, Union

# Initialize the OpenAI client with Vocareum configuration
client = OpenAI(
    base_url="https://openai.vocareum.com/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)

def safe_openai_call(messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo", temperature: float = 0.7) -> Dict[str, Any]:
    """
    Makes a safe call to the OpenAI API with error handling.

    Args:
        messages (List[Dict[str, str]]): The messages to send to the API
        model (str): The model to use
        temperature (float): The temperature to use for the API call

    Returns:
        Dict[str, Any]: Either the API response or an error message
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        return {
            "success": True,
            "content": response.choices[0].message.content,
            "response": response
        }
    except BadRequestError as e:
        if "Insufficient budget available" in str(e):
            error_msg = "OpenAI API budget exceeded. Please check your Vocareum account."
            print(f"Error: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "content": "I'm unable to process this request due to API budget limitations."
            }
        else:
            error_msg = f"OpenAI API error: {str(e)}"
            print(f"Error: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "content": "I encountered an error while processing your request."
            }
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"Error: {error_msg}")
        return {
            "success": False,
            "error": error_msg,
            "content": "I encountered an unexpected error while processing your request."
        }

class DirectPromptAgent:
    """
    A simple agent that sends a prompt directly to the LLM and returns the response.
    """

    def __init__(self, system_prompt: str = "You are a helpful assistant."):
        """
        Initialize the DirectPromptAgent.

        Args:
            system_prompt (str): The system prompt to use for the agent.
        """
        self.system_prompt = system_prompt

    def respond(self, prompt: str) -> str:
        """
        Respond to the given prompt.

        Args:
            prompt (str): The user prompt to send to the LLM.

        Returns:
            str: The LLM's response text.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]

        result = safe_openai_call(messages)
        return result["content"]

    def run(self, prompt: str) -> str:
        """
        Legacy method for backward compatibility.

        Args:
            prompt (str): The user prompt to send to the LLM.

        Returns:
            str: The LLM's response.
        """
        return self.respond(prompt)


class AugmentedPromptAgent:
    """
    An agent that augments the user prompt with additional instructions or context.
    """

    def __init__(self, system_prompt: str = "You are a helpful assistant.", 
                 prompt_prefix: str = "", 
                 prompt_suffix: str = ""):
        """
        Initialize the AugmentedPromptAgent.

        Args:
            system_prompt (str): The system prompt to use for the agent.
            prompt_prefix (str): Text to add before the user prompt.
            prompt_suffix (str): Text to add after the user prompt.
        """
        self.system_prompt = system_prompt
        self.prompt_prefix = prompt_prefix
        self.prompt_suffix = prompt_suffix

    def respond(self, prompt: str) -> str:
        """
        Respond to the given prompt, augmented with prefix and suffix.

        Args:
            prompt (str): The user prompt to send to the LLM.

        Returns:
            str: The LLM's response text.
        """
        augmented_prompt = f"{self.prompt_prefix}{prompt}{self.prompt_suffix}"

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": augmented_prompt}
        ]

        result = safe_openai_call(messages)
        return result["content"]

    def run(self, prompt: str) -> str:
        """
        Legacy method for backward compatibility.

        Args:
            prompt (str): The user prompt to send to the LLM.

        Returns:
            str: The LLM's response.
        """
        return self.respond(prompt)


class KnowledgeAugmentedPromptAgent:
    """
    An agent that augments the prompt with specific knowledge or context.
    """

    def __init__(self, system_prompt: str = "You are a helpful assistant.", 
                 knowledge: str = ""):
        """
        Initialize the KnowledgeAugmentedPromptAgent.

        Args:
            system_prompt (str): The system prompt to use for the agent.
            knowledge (str): The knowledge or context to include with every prompt.
        """
        self.system_prompt = system_prompt
        self.knowledge = knowledge

    def respond(self, prompt: str) -> str:
        """
        Respond to the given prompt, augmented with knowledge.

        Args:
            prompt (str): The user prompt to send to the LLM.

        Returns:
            str: The LLM's response text.
        """
        augmented_prompt = f"Using the following information:\n\n{self.knowledge}\n\nRespond to: {prompt}"

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": augmented_prompt}
        ]

        result = safe_openai_call(messages)
        return result["content"]

    def run(self, prompt: str) -> str:
        """
        Legacy method for backward compatibility.

        Args:
            prompt (str): The user prompt to send to the LLM.

        Returns:
            str: The LLM's response.
        """
        return self.respond(prompt)


class RAGKnowledgePromptAgent:
    """
    An agent that uses Retrieval-Augmented Generation to dynamically retrieve
    relevant knowledge before responding to a prompt.
    """

    def __init__(self, system_prompt: str = "You are a helpful assistant.", 
                 knowledge_base: Dict[str, str] = None):
        """
        Initialize the RAGKnowledgePromptAgent.

        Args:
            system_prompt (str): The system prompt to use for the agent.
            knowledge_base (Dict[str, str]): A dictionary mapping topics to knowledge content.
        """
        self.system_prompt = system_prompt
        self.knowledge_base = knowledge_base or {}

    def retrieve_knowledge(self, prompt: str) -> str:
        """
        Retrieve relevant knowledge based on the prompt.

        Args:
            prompt (str): The user prompt.

        Returns:
            str: The retrieved knowledge.
        """
        # Simple keyword-based retrieval for demonstration
        # In a real implementation, this would use embeddings and semantic search
        retrieved_knowledge = []

        for topic, content in self.knowledge_base.items():
            if any(keyword in prompt.lower() for keyword in topic.lower().split()):
                retrieved_knowledge.append(f"Topic: {topic}\n{content}")

        if not retrieved_knowledge:
            return "No specific knowledge found for this query."

        return "\n\n".join(retrieved_knowledge)

    def respond(self, prompt: str) -> str:
        """
        Respond to the given prompt, augmented with retrieved knowledge.

        Args:
            prompt (str): The user prompt to send to the LLM.

        Returns:
            str: The LLM's response text.
        """
        retrieved_knowledge = self.retrieve_knowledge(prompt)
        augmented_prompt = f"Using the following information:\n\n{retrieved_knowledge}\n\nRespond to: {prompt}"

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": augmented_prompt}
        ]

        result = safe_openai_call(messages)
        return result["content"]

    def run(self, prompt: str) -> str:
        """
        Legacy method for backward compatibility.

        Args:
            prompt (str): The user prompt to send to the LLM.

        Returns:
            str: The LLM's response.
        """
        return self.respond(prompt)


class EvaluationAgent:
    """
    An agent that evaluates responses or outputs based on specific criteria.
    """

    def __init__(self, system_prompt: str = "You are a critical evaluator. Your job is to evaluate responses based on accuracy, completeness, and relevance."):
        """
        Initialize the EvaluationAgent.

        Args:
            system_prompt (str): The system prompt to use for the agent.
        """
        self.system_prompt = system_prompt

    def evaluate(self, prompt: str, response: str, criteria: List[str] = None, max_iterations: int = 3) -> Dict[str, Any]:
        """
        Evaluate a response based on the given criteria with iterative improvement.

        Args:
            prompt (str): The original prompt.
            response (str): The response to evaluate.
            criteria (List[str]): Specific criteria to evaluate against.
            max_iterations (int): Maximum number of iterations for evaluation.

        Returns:
            Dict[str, Any]: Dictionary with final_response, evaluation, and iteration_count.
        """
        criteria_str = "\n".join([f"- {c}" for c in (criteria or ["Accuracy", "Completeness", "Relevance"])])
        current_response = response
        iteration_count = 0

        while iteration_count < max_iterations:
            evaluation_prompt = f"""
            Original Prompt: {prompt}

            Response to Evaluate: {current_response}

            Please evaluate the response based on the following criteria:
            {criteria_str}

            For each criterion, provide a score from 1-10 and brief feedback.
            Then provide an overall score and summary of the evaluation.

            If the response needs improvement, provide specific correction instructions.

            Format your response as a JSON object with the following structure:
            {{
                "criteria": {{
                    "criterion1": {{
                        "score": X,
                        "feedback": "Your feedback here"
                    }},
                    ...
                }},
                "overall": {{
                    "score": X,
                    "summary": "Your summary here"
                }},
                "needs_improvement": true/false,
                "correction_instructions": "Specific instructions for improvement if needed"
            }}
            """

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": evaluation_prompt}
            ]

            # Use temperature=0 for more consistent evaluations
            result = safe_openai_call(messages, temperature=0)
            evaluation_text = result["content"]

            # Try to parse the evaluation as JSON
            try:
                evaluation = json.loads(evaluation_text)

                # Check if the response needs improvement
                if not evaluation.get("needs_improvement", False) or iteration_count >= max_iterations - 1:
                    # Return the final evaluation
                    return {
                        "final_response": current_response,
                        "evaluation": evaluation,
                        "iteration_count": iteration_count + 1
                    }

                # Get correction instructions
                correction_instructions = evaluation.get("correction_instructions", "")

                # Generate improved response
                improvement_prompt = f"""
                Original Prompt: {prompt}

                Current Response: {current_response}

                Evaluation: {json.dumps(evaluation, indent=2)}

                Correction Instructions: {correction_instructions}

                Please provide an improved response that addresses the correction instructions.
                """

                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": improvement_prompt}
                ]

                improvement_result = safe_openai_call(messages)
                current_response = improvement_result["content"]
                iteration_count += 1

            except json.JSONDecodeError:
                # If we can't parse the evaluation as JSON, return what we have
                return {
                    "final_response": current_response,
                    "evaluation": evaluation_text,
                    "iteration_count": iteration_count + 1
                }

        # If we've reached the maximum number of iterations, return the current response
        return {
            "final_response": current_response,
            "evaluation": evaluation_text,
            "iteration_count": iteration_count
        }


class RoutingAgent:
    """
    An agent that routes prompts to different handlers based on content analysis.
    """

    def __init__(self, system_prompt: str = "You are a routing assistant. Your job is to analyze requests and route them to the appropriate handler."):
        """
        Initialize the RoutingAgent.

        Args:
            system_prompt (str): The system prompt to use for the agent.
        """
        self.system_prompt = system_prompt
        self.routes = {}

    def add_route(self, name: str, description: str, handler: Any):
        """
        Add a route to the routing agent.

        Args:
            name (str): The name of the route.
            description (str): Description of what this route handles.
            handler (Any): The handler function or object for this route.
        """
        self.routes[name] = {
            "description": description,
            "handler": handler
        }

    def route(self, prompt: str) -> Dict[str, Any]:
        """
        Determine the appropriate route for a given prompt using embeddings and cosine similarity.

        Args:
            prompt (str): The user prompt to route.

        Returns:
            Dict[str, Any]: The routing decision including route name and confidence.
        """
        # Import utils here to avoid circular imports
        from phase_2.utils import get_embedding, cosine_similarity

        # Check if agents attribute exists and is not empty
        if not hasattr(self, 'agents') or not self.agents:
            if not self.routes:
                return {"route": None, "confidence": 0, "explanation": "No routes configured"}

            # Fall back to traditional routing if agents attribute is not set
            routes_description = "\n".join([f"- {name}: {route['description']}" for name, route in self.routes.items()])

            routing_prompt = f"""
            User Request: {prompt}

            Available Routes:
            {routes_description}

            Analyze the user request and determine the most appropriate route.
            Format your response as a JSON object with the following structure:
            {{
                "route": "route_name",
                "confidence": X,  # A number between 0 and 1
                "explanation": "Your explanation here"
            }}

            If none of the routes seem appropriate, set route to null and explain why.
            """

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": routing_prompt}
            ]

            result = safe_openai_call(messages)
            return result["content"]

        # Get embedding for the prompt
        prompt_embedding = get_embedding(prompt)

        # Calculate similarity scores for each agent
        best_route = None
        best_score = -1
        best_explanation = ""

        for agent in self.agents:
            # Create a description that combines the agent's name, description, and expertise
            agent_description = f"{agent['name']}: {agent['description']} Expertise: {', '.join(agent['expertise'])}"

            # Get embedding for the agent description
            agent_embedding = get_embedding(agent_description)

            # Calculate similarity
            similarity = cosine_similarity(prompt_embedding, agent_embedding)

            if similarity > best_score:
                best_score = similarity
                best_route = agent['name']
                best_explanation = f"This request matches the expertise of {agent['name']} with a similarity score of {similarity:.2f}."

        # Format the result as a JSON string
        result = {
            "route": best_route,
            "confidence": best_score,
            "explanation": best_explanation
        }

        return json.dumps(result)

    def respond(self, prompt: str) -> Dict[str, Any]:
        """
        Respond to the given prompt by routing it to the appropriate handler.

        Args:
            prompt (str): The user prompt to route.

        Returns:
            Dict[str, Any]: The routing decision including route name and confidence.
        """
        return self.route(prompt)

    def process(self, prompt: str):
        """
        Process a prompt by routing it to the appropriate handler.

        Args:
            prompt (str): The user prompt to process.

        Returns:
            Any: The result from the handler.
        """
        routing_result = self.route(prompt)

        # Parse the routing result
        try:
            routing_data = json.loads(routing_result) if isinstance(routing_result, str) else routing_result
            route_name = routing_data.get("route", "default")
        except (json.JSONDecodeError, AttributeError):
            # If parsing fails, use default route
            route_name = "default"

        # Check if we have agents with func attribute
        if hasattr(self, 'agents') and self.agents:
            for agent in self.agents:
                if agent.get('name') == route_name and 'func' in agent:
                    return agent['func'](prompt)

        # Fall back to traditional routes
        if route_name in self.routes:
            return self.routes[route_name]["handler"](prompt)
        else:
            return f"No handler found for route: {route_name}"


class ActionPlanningAgent:
    """
    An agent that plans and executes a sequence of actions to accomplish a task.
    """

    def __init__(self, system_prompt: str = "You are an action planning assistant. Your job is to break down tasks into actionable steps."):
        """
        Initialize the ActionPlanningAgent.

        Args:
            system_prompt (str): The system prompt to use for the agent.
        """
        self.system_prompt = system_prompt
        self.actions = {}

    def add_action(self, name: str, description: str, handler: Any):
        """
        Add an action to the agent's repertoire.

        Args:
            name (str): The name of the action.
            description (str): Description of what this action does.
            handler (Any): The handler function or object for this action.
        """
        self.actions[name] = {
            "description": description,
            "handler": handler
        }

    def plan(self, task: str) -> List[Dict[str, Any]]:
        """
        Create a plan of actions for a given task.

        Args:
            task (str): The task to plan for.

        Returns:
            List[Dict[str, Any]]: A list of planned actions with parameters.
        """
        if not self.actions:
            return [{"error": "No actions configured"}]

        actions_description = "\n".join([f"- {name}: {action['description']}" for name, action in self.actions.items()])

        planning_prompt = f"""
        Task: {task}

        Available Actions:
        {actions_description}

        Create a plan to accomplish this task using the available actions.
        For each step, specify the action name and any required parameters.
        Format your response as a JSON array with the following structure:
        [
            {{
                "action": "action_name",
                "parameters": {{
                    "param1": "value1",
                    ...
                }},
                "explanation": "Why this action is needed"
            }},
            ...
        ]
        """

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": planning_prompt}
        ]

        result = safe_openai_call(messages)

        # In a real implementation, we would parse the JSON response
        # For simplicity, we're returning the raw response
        return result["content"]

    def respond(self, task: str) -> Dict[str, Any]:
        """
        Respond to the given task by planning and executing actions.

        Args:
            task (str): The task to accomplish.

        Returns:
            Dict[str, Any]: The results of the execution.
        """
        return self.run(task)

    def execute_plan(self, plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute a plan of actions.

        Args:
            plan (List[Dict[str, Any]]): The plan to execute.

        Returns:
            List[Dict[str, Any]]: Results of each action in the plan.
        """
        # This is a placeholder implementation
        # In a real scenario, you would parse the plan and execute each action
        results = []

        for step in plan:
            action_name = step.get("action")
            parameters = step.get("parameters", {})

            if action_name in self.actions:
                try:
                    result = self.actions[action_name]["handler"](**parameters)
                    results.append({
                        "action": action_name,
                        "success": True,
                        "result": result
                    })
                except Exception as e:
                    results.append({
                        "action": action_name,
                        "success": False,
                        "error": str(e)
                    })
            else:
                results.append({
                    "action": action_name,
                    "success": False,
                    "error": f"Unknown action: {action_name}"
                })

        return results

    def run(self, task: str) -> Dict[str, Any]:
        """
        Plan and execute a task.

        Args:
            task (str): The task to accomplish.

        Returns:
            Dict[str, Any]: The results of the execution.
        """
        plan_response = self.plan(task)

        # Try to parse the plan
        try:
            plan = json.loads(plan_response) if isinstance(plan_response, str) else plan_response
        except (json.JSONDecodeError, AttributeError):
            # If parsing fails, use an empty plan
            plan = []

        results = self.execute_plan(plan)

        return {
            "task": task,
            "plan": plan_response,
            "results": results
        }
