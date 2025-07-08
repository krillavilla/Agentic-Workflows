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

def safe_openai_call(messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo") -> Dict[str, Any]:
    """
    Makes a safe call to the OpenAI API with error handling.

    Args:
        messages (List[Dict[str, str]]): The messages to send to the API
        model (str): The model to use

    Returns:
        Dict[str, Any]: Either the API response or an error message
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages
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

    def run(self, prompt: str) -> str:
        """
        Run the agent with the given prompt.

        Args:
            prompt (str): The user prompt to send to the LLM.

        Returns:
            str: The LLM's response.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]

        result = safe_openai_call(messages)
        return result["content"]


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

    def run(self, prompt: str) -> str:
        """
        Run the agent with the given prompt, augmented with prefix and suffix.

        Args:
            prompt (str): The user prompt to send to the LLM.

        Returns:
            str: The LLM's response.
        """
        augmented_prompt = f"{self.prompt_prefix}{prompt}{self.prompt_suffix}"

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": augmented_prompt}
        ]

        result = safe_openai_call(messages)
        return result["content"]


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

    def run(self, prompt: str) -> str:
        """
        Run the agent with the given prompt, augmented with knowledge.

        Args:
            prompt (str): The user prompt to send to the LLM.

        Returns:
            str: The LLM's response.
        """
        augmented_prompt = f"Using the following information:\n\n{self.knowledge}\n\nRespond to: {prompt}"

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": augmented_prompt}
        ]

        result = safe_openai_call(messages)
        return result["content"]


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

    def run(self, prompt: str) -> str:
        """
        Run the agent with the given prompt, augmented with retrieved knowledge.

        Args:
            prompt (str): The user prompt to send to the LLM.

        Returns:
            str: The LLM's response.
        """
        retrieved_knowledge = self.retrieve_knowledge(prompt)
        augmented_prompt = f"Using the following information:\n\n{retrieved_knowledge}\n\nRespond to: {prompt}"

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": augmented_prompt}
        ]

        result = safe_openai_call(messages)
        return result["content"]


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

    def evaluate(self, prompt: str, response: str, criteria: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate a response based on the given criteria.

        Args:
            prompt (str): The original prompt.
            response (str): The response to evaluate.
            criteria (List[str]): Specific criteria to evaluate against.

        Returns:
            Dict[str, Any]: Evaluation results including scores and feedback.
        """
        criteria_str = "\n".join([f"- {c}" for c in (criteria or ["Accuracy", "Completeness", "Relevance"])])

        evaluation_prompt = f"""
        Original Prompt: {prompt}

        Response to Evaluate: {response}

        Please evaluate the response based on the following criteria:
        {criteria_str}

        For each criterion, provide a score from 1-10 and brief feedback.
        Then provide an overall score and summary of the evaluation.
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
            }}
        }}
        """

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": evaluation_prompt}
        ]

        result = safe_openai_call(messages)

        # In a real implementation, we would parse the JSON response
        # For simplicity, we're returning the raw response
        return result["content"]


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
        Determine the appropriate route for a given prompt.

        Args:
            prompt (str): The user prompt to route.

        Returns:
            Dict[str, Any]: The routing decision including route name and confidence.
        """
        if not self.routes:
            return {"route": None, "confidence": 0, "explanation": "No routes configured"}

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

        # In a real implementation, we would parse the JSON response
        # For simplicity, we're returning the raw response
        return result["content"]

    def process(self, prompt: str):
        """
        Process a prompt by routing it to the appropriate handler.

        Args:
            prompt (str): The user prompt to process.

        Returns:
            Any: The result from the handler.
        """
        routing_result = self.route(prompt)
        # In a real implementation, we would parse the JSON response
        # For simplicity, we're assuming the format and extracting the route

        # This is a placeholder implementation
        # In a real scenario, you would parse the JSON and get the route name
        route_name = "default"  # Placeholder

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
        # In a real implementation, we would parse the JSON response

        # This is a placeholder implementation
        # In a real scenario, you would parse the JSON and get the plan
        plan = []  # Placeholder

        results = self.execute_plan(plan)

        return {
            "task": task,
            "plan": plan_response,
            "results": results
        }
