# Agentic-Workflows

A Python project implementing agentic workflows for automated task processing using OpenAI's GPT models.

## Project Overview

This project demonstrates the implementation of agentic systems that can process and respond to product specification emails. It is divided into two phases:

1. **Phase 1**: Implementation of various agent classes that serve as building blocks for agentic workflows
2. **Phase 2**: Implementation of a complete agentic workflow that routes product specification emails to the appropriate team member and generates responses

## Project Structure

```
Agentic-Workflows/
├── phase_1/
│   ├── workflow_agents/
│   │   ├── __init__.py
│   │   └── base_agents.py
│   ├── direct_prompt_agent_test.py
│   ├── augmented_prompt_agent_test.py
│   ├── knowledge_augmented_prompt_agent_test.py
│   ├── rag_knowledge_prompt_agent_test.py
│   ├── evaluation_agent_test.py
│   ├── routing_agent_test.py
│   └── action_planning_agent_test.py
├── phase_2/
│   ├── agentic_workflow.py
│   ├── Product-Spec-Email-Router.txt
│└── README.md
```

## Phase 1: Agent Library

Phase 1 implements seven agent classes in `workflow_agents/base_agents.py`:

1. **DirectPromptAgent**: A simple agent that sends a prompt directly to the LLM and returns the response
2. **AugmentedPromptAgent**: An agent that augments the user prompt with additional instructions or context
3. **KnowledgeAugmentedPromptAgent**: An agent that augments the prompt with specific knowledge or context
4. **RAGKnowledgePromptAgent**: An agent that uses Retrieval-Augmented Generation to dynamically retrieve relevant knowledge
5. **EvaluationAgent**: An agent that evaluates responses or outputs based on specific criteria
6. **RoutingAgent**: An agent that routes prompts to different handlers based on content analysis
7. **ActionPlanningAgent**: An agent that plans and executes a sequence of actions to accomplish a task

Each agent has a corresponding test script that demonstrates its functionality.

## Phase 2: Agentic Workflow

Phase 2 implements a complete agentic workflow in `agentic_workflow.py` that:

1. Routes product specification emails to the appropriate team member (Product Manager, Program Manager, or Development Engineer)
2. Generates responses using specialized agents for each role
3. Evaluates the quality of the responses

The routing logic is defined in `Product-Spec-Email-Router.txt`.

## Setup and Installation

### Prerequisites

- Python 3.7 or higher
- OpenAI API key

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/Agentic-Workflows.git
   cd Agentic-Workflows
   ```

2. Install the required packages:
   ```
   pip install openai python-dotenv
   ```

3. Set up your OpenAI API key:
   - Create a copy of the `.env` file in the `phase_2` directory
   - Replace `your-api-key-here` with your actual OpenAI API key

## Usage

### Testing Individual Agents (Phase 1)

To test individual agents, run the corresponding test script:

```
cd Agentic-Workflows
python phase_1/direct_prompt_agent_test.py
python phase_1/augmented_prompt_agent_test.py
python phase_1/knowledge_augmented_prompt_agent_test.py
python phase_1/rag_knowledge_prompt_agent_test.py
python phase_1/evaluation_agent_test.py
python phase_1/routing_agent_test.py
python phase_1/action_planning_agent_test.py
```

### Running the Agentic Workflow (Phase 2)

To run the complete agentic workflow:

```
cd Agentic-Workflows
python phase_2/agentic_workflow.py
```

This will process three example emails and demonstrate the routing, response generation, and evaluation steps of the workflow.

### Detailed Testing Guide

For comprehensive instructions on how to test the functionality of this project, please refer to the [Testing Guide](TESTING_GUIDE.md). This guide includes:

- Detailed steps for testing each agent
- Expected outputs and how to verify them
- Troubleshooting common issues
- Advanced testing options

## Customization

### Adding New Agents

To add a new agent type:

1. Add the agent class to `phase_1/workflow_agents/base_agents.py`
2. Create a test script for the new agent
3. Update the agentic workflow to use the new agent if needed

### Modifying Routing Logic

To modify the routing logic:

1. Edit the `phase_2/Product-Spec-Email-Router.txt` file
2. Update the routing rules and examples as needed

### Extending the Workflow

To extend the agentic workflow:

1. Add new actions to the `ActionPlanningAgent` in `phase_2/agentic_workflow.py`
2. Modify the `process_email` function to include additional steps
3. Add new specialized agents for different roles or tasks

## Example Output

When running the agentic workflow, you'll see output similar to:

```
Processing Product Manager Email:
Starting email processing workflow...
--------------------------------------------------
Loaded routing knowledge
Step 1: Routing the email...
Routing result: {"route": "product_manager", "confidence": 0.95, "explanation": "This email is about feature prioritization for a product release, which is a core responsibility of the Product Manager."}
--------------------------------------------------
Step 2: Generating response using the product_manager agent...
Generated response: [Response from the Product Manager agent]
--------------------------------------------------
Step 3: Evaluating the response...
Evaluation result: [Evaluation of the response]
--------------------------------------------------

[Similar output for Program Manager and Development Engineer emails]
```

## License

This project is licensed under the MIT License.
