# ADK Agent Studio

> A comprehensive development environment for Google ADK (Agent Development Kit)

Build, test, and deploy production-ready AI agents with an interactive UI, config-based creation, and zero boilerplate. Features include hierarchical agent structures, tool integration, sequential workflows, structured output, and instant code export.

## Features

- **Configuration-driven**: Define agents using JSON or YAML files
- **Multiple agents**: Easily create and manage different agents with unique behaviors
- **Session management**: Automatic session and user ID generation for each agent instance
- **Async/await support**: Modern async Python for efficient I/O operations
- **Clean API**: Simple, intuitive interface for agent interactions
- **Web UI**: Interactive Streamlit interface for testing and visualizing agent execution
- **Structured Output**: Define JSON schemas for consistent, parseable agent responses
- **Tools System**: Support for both function-based and class-based tools
- **Sub-Agents**: Hierarchical agent structures with delegation capabilities
- **Sequential Workflows**: Multi-step pipelines with state passing between agents

## Installation

1. Ensure you have Python 3.8+ installed
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env` file (if needed)

## Quick Start

### Option 1: Web UI (Recommended)

The easiest way to get started is using the interactive web interface:

```bash
# Windows
run_ui.bat

# Unix/Mac
bash run_ui.sh

# Or directly
streamlit run app.py
```

Then open your browser to `http://localhost:8501` and you'll see:
- 📋 List of all available agents
- 💬 Chat mode with conversation history
- 🎯 Single execution for testing queries
- 📊 Agent metadata and configuration viewer
- ➕ Agent creation form with AI assistance
- 📤 Code export functionality
- 🔍 Execution details and session info

### Option 2: Python Code

```python
import asyncio
from agent_def import AgentManager

async def main():
    # Create agent from config file
    agent = AgentManager("configs/template_agent.json")
    
    # Initialize the agent
    await agent.initialize()
    
    # Send a message and get response
    response = await agent.send_message("Hello! How can you help me?")
    print(response)
    
    # Cleanup
    await agent.close()

if __name__ == "__main__":
    asyncio.run(main())
```

### Running the Examples

The project includes example scripts demonstrating different use cases:

```bash
# Run the basic agent example
python agent_def/agent.py

# Run comprehensive examples
python example_usage.py
```

## Configuration

Agents are configured using JSON or YAML files stored in the `configs/` directory.

### Example Configuration (`configs/template_agent.json`)

```json
{
  "app_name": "template_agent",
  "agent_type": "llm",
  "agent": {
    "name": "template_agent",
    "model": "gemini-2.5-flash",
    "description": "The template agent is a template for creating new agents.",
    "instruction": "You are a template agent. You are used to create new agents."
  },
  "tools": []
}
```

### Configuration Fields

- **app_name**: Unique identifier for the application/agent
- **agent_type**: Type of agent - `"llm"` (default, intelligent) or `"sequential"` (workflow)
- **agent.name**: Display name for the agent
- **agent.model**: The AI model to use (e.g., "gemini-2.5-flash") - required for LLM agents
- **agent.description**: Brief description of the agent's purpose
- **agent.instruction**: System instructions that define the agent's behavior - required for LLM agents
- **agent.response_schema**: JSON schema defining structured output format (optional, see Structured Output section)
- **agent.output_key**: Key name for storing output in shared state (optional, for sequential workflows)
- **tools**: List of tool names to load (optional, see Tools section below)
- **sub_agents**: List of sub-agent names (optional for LLM agents, required for sequential)

## Tools

Agents can be enhanced with tools to perform specific actions like calculations, web searches, database queries, and more.

### Tool Types

The system supports two types of tools:

1. **Function-based tools**: Simple async functions for stateless operations
2. **Class-based tools**: Classes inheriting from `BaseTool` for complex operations with state

### Using Tools

To enable tools for an agent, add them to the `tools` array in your config:

```json
{
  "app_name": "assistant_with_tools",
  "agent_type": "llm",
  "agent": {
    "name": "helpful_assistant",
    "model": "gemini-2.5-flash",
    "description": "An assistant with calculator and search capabilities",
    "instruction": "You are a helpful assistant. Use tools when appropriate."
  },
  "tools": [
    "calculator",
    "WebSearchTool"
  ]
}
```

### Built-in Tools

#### `calculator` (Function-based)
- **Purpose**: Evaluate mathematical expressions safely
- **Example**: `calculator(expression="2 + 2 * 5")`
- **Config**: None needed

#### `WebSearchTool` (Class-based)
- **Purpose**: Search the web for current information
- **Parameters**: 
  - `query` (required): Search query string
  - `num_results` (optional): Number of results (1-10, default: 3)
- **Environment Variables**:
  - `SEARCH_API_KEY`: API key for search service (optional for demo mode)
  - `SEARCH_ENGINE`: Search engine to use (default: "demo")

### Creating Custom Tools

#### Simple Function Tool

Create a file in `tools/` directory:

```python
# tools/my_tool.py
async def my_tool(param1: str, param2: int) -> str:
    """
    Brief description of what the tool does.
    
    Args:
        param1: Description of first parameter
        param2: Description of second parameter
    
    Returns:
        Result as a string
    """
    # Your tool logic here
    return f"Processed {param1} with {param2}"
```

Then add `"my_tool"` to the `tools` array in your config.

#### Complex Class-Based Tool

```python
# tools/my_advanced_tool.py
import os
from agent_def.base_tool import BaseTool, ToolParameter

class MyAdvancedTool(BaseTool):
    """Tool with state management and validation."""
    
    @property
    def name(self) -> str:
        return "my_advanced_tool"
    
    @property
    def description(self) -> str:
        return "Description of what this tool does"
    
    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="input_text",
                type="string",
                description="Text to process",
                required=True
            ),
            ToolParameter(
                name="option",
                type="string",
                description="Processing option",
                required=False,
                default="default"
            )
        ]
    
    def _setup(self):
        """Load config from environment variables."""
        self.api_key = os.getenv("MY_API_KEY")
        print("MyAdvancedTool initialized")
    
    async def execute(self, **kwargs) -> str:
        """Execute the tool."""
        params = self.validate_parameters(**kwargs)
        # Your tool logic here
        return f"Processed: {params['input_text']}"
    
    def _teardown(self):
        """Cleanup resources."""
        print("MyAdvancedTool cleaned up")
```

Then add `"MyAdvancedTool"` to the `tools` array in your config.

### Tool Naming Convention

- **Function tools**: Use lowercase with underscores (e.g., `calculator`, `send_email`)
  - File: `tools/calculator.py` → Function: `calculator()`
- **Class tools**: Use PascalCase (e.g., `WebSearchTool`, `DatabaseQueryTool`)
  - File: `tools/web_search_tool.py` → Class: `WebSearchTool`

### Tool Configuration

Tools get all configuration from environment variables, not from the config file. Set environment variables in your `.env` file:

```bash
# .env
SEARCH_API_KEY=your_api_key_here
SEARCH_ENGINE=google
DATABASE_URL=postgresql://localhost/mydb
```

## Sub-Agents

Sub-agents allow you to create hierarchical agent systems where a coordinator agent delegates specific tasks to specialized sub-agents. This is powerful for:
- **Specialization**: Each sub-agent focuses on a specific domain
- **Modularity**: Sub-agents can be reused across different coordinator agents
- **Scalability**: Complex tasks can be broken down and distributed

### How Sub-Agents Work

- Sub-agents share the same session as their parent agent
- They can have their own tools and even their own sub-agents (nested hierarchy)
- The coordinator agent decides when to delegate to a sub-agent
- Sub-agents are referenced by name in the config

### Using Sub-Agents

To create an agent hierarchy, add a `sub_agents` array to your config:

```json
{
  "app_name": "coordinator_agent",
  "agent_type": "llm",
  "agent": {
    "name": "coordinator_agent",
    "model": "gemini-2.5-flash",
    "description": "Main coordinator that delegates to specialists",
    "instruction": "You coordinate a team. Delegate greetings to 'greeting_agent' and farewells to 'farewell_agent'. Handle other requests yourself."
  },
  "tools": ["get_weather"],
  "sub_agents": [
    "greeting_agent",
    "farewell_agent"
  ]
}
```

### Creating Sub-Agent Configs

Sub-agents have their own config files in the `configs/` directory:

**configs/greeting_agent.json:**
```json
{
  "app_name": "greeting_specialist",
  "agent_type": "llm",
  "agent": {
    "name": "greeting_agent",
    "model": "gemini-2.5-flash",
    "description": "Handles greetings and welcome messages",
    "instruction": "You specialize in friendly greetings. Respond warmly to 'Hi', 'Hello', etc."
  }
}
```

**configs/farewell_agent.json:**
```json
{
  "app_name": "farewell_specialist",
  "agent_type": "llm",
  "agent": {
    "name": "farewell_agent",
    "model": "gemini-2.5-flash",
    "description": "Handles farewells and goodbye messages",
    "instruction": "You specialize in farewells. Respond kindly to 'Bye', 'Goodbye', etc."
  }
}
```

### Example: Coordinator with Sub-Agents

```python
from agent_def import AgentManager

# Create coordinator agent (automatically loads sub-agents)
coordinator = AgentManager("configs/coordinator_agent.json")
await coordinator.initialize()

# Greetings are delegated to greeting_agent
response = await coordinator.send_message("Hello!")
# -> "Hi there! Glad to greet you!"

# Weather queries use the coordinator's own tool
response = await coordinator.send_message("What's the weather in London?")
# -> "The weather in London is Cloudy, 15°C..."

# Farewells are delegated to farewell_agent
response = await coordinator.send_message("Goodbye!")
# -> "Goodbye! Take care and have a wonderful day!"
```

### Nested Sub-Agents

Sub-agents can have their own sub-agents, creating multi-level hierarchies:

```json
{
  "agent_type": "llm",
  "agent": {
    "name": "executive_agent",
    "model": "gemini-2.5-flash",
    "description": "Executive level coordinator",
    "instruction": "Delegate tasks to manager_agent..."
  },
  "sub_agents": ["manager_agent"]
}
```

```json
{
  "agent_type": "llm",
  "agent": {
    "name": "manager_agent",
    "model": "gemini-2.5-flash",
    "description": "Middle management coordinator",
    "instruction": "Delegate work to worker agents..."
  },
  "sub_agents": ["worker_agent_1", "worker_agent_2"]
}
```

### Sub-Agent Naming Convention

Sub-agent config files should be named: `{agent_name}.json`

If your sub-agent is named `greeting_agent`, the config should be at `configs/greeting_agent.json`.

### Best Practices

1. **Clear Instructions**: Tell the coordinator agent exactly when to delegate to each sub-agent
2. **Focused Sub-Agents**: Each sub-agent should have a specific, well-defined purpose
3. **Tool Distribution**: Give tools to the agent that will use them most
4. **Session Sharing**: Remember that sub-agents share the session, so they have access to conversation history

## Sequential Agents (Workflows)

In addition to LLM-based agents that use intelligence to decide what to do, the system supports **SequentialAgents** for workflow-oriented pipelines where sub-agents run in a predefined order.

### LLM Agent vs Sequential Agent

| Feature | LLM Agent (default) | Sequential Agent |
|---------|-------------------|------------------|
| Decision Making | LLM decides when to delegate | Runs sub-agents in order |
| Use Case | Dynamic, intelligent routing | Fixed workflows, pipelines |
| Sub-Agents | Optional, delegated as needed | Required, runs all in sequence |
| Tools | Can have tools | Typically no tools (orchestration only) |
| Instructions | Needs detailed instructions | No instructions needed |

### Creating a Sequential Agent

Sequential agents are perfect for pipelines like code review workflows, data processing chains, or multi-step analysis.

**Example: Code Pipeline**

```json
{
  "app_name": "code_pipeline",
  "agent_type": "sequential",
  "agent": {
    "name": "CodePipelineAgent",
    "description": "Executes a sequence of code writing, reviewing, and refactoring."
  },
  "sub_agents": [
    "code_writer_agent",
    "code_reviewer_agent_sequential",
    "code_refactorer_agent"
  ]
}
```

**Sub-Agent Configs:**

```json
// configs/code_writer_agent.json
{
  "app_name": "code_writer",
  "agent_type": "llm",
  "agent": {
    "name": "code_writer_agent",
    "model": "gemini-2.5-flash",
    "description": "Writes code based on requirements",
    "instruction": "Write clean code. Output ONLY the code.",
    "output_key": "generated_code"
  }
}

// configs/code_reviewer_agent_sequential.json
{
  "app_name": "code_reviewer_sequential",
  "agent_type": "llm",
  "agent": {
    "name": "code_reviewer_agent",
    "model": "gemini-2.5-flash",
    "description": "Reviews code and provides feedback",
    "instruction": "Review this code:\n```python\n{generated_code}\n```\nProvide feedback.",
    "output_key": "review_comments"
  }
}

// configs/code_refactorer_agent.json
{
  "app_name": "code_refactorer",
  "agent_type": "llm",
  "agent": {
    "name": "code_refactorer_agent",
    "model": "gemini-2.5-flash",
    "description": "Refactors code based on review feedback",
    "instruction": "Original:\n{generated_code}\nReview:\n{review_comments}\nRefactor it.",
    "output_key": "refactored_code"
  }
}
```

### Using a Sequential Agent

```python
from agent_def import AgentManager

# Create sequential pipeline
pipeline = AgentManager("configs/code_pipeline_agent.json")
await pipeline.initialize()

# Send a request - it flows through all agents in order
response = await pipeline.send_message("""
Write a Python function to calculate factorial.
It should handle edge cases and be efficient.
""")

# Output will be the result after going through:
# 1. code_writer_agent (writes initial code)
# 2. code_reviewer_agent_sequential (reviews it)
# 3. code_refactorer_agent (refactors based on feedback)
```

### When to Use Sequential Agents

✅ **Use Sequential Agents for:**
- Fixed workflows (code review, data pipelines)
- Multi-step processing where order matters
- Quality assurance chains (generate → review → improve)
- Document processing pipelines (extract → analyze → summarize)

❌ **Use LLM Agents (default) for:**
- Dynamic decision-making
- Task routing based on content
- Flexible delegation
- General-purpose assistants

### State Passing Between Agents

In sequential workflows, agents often need to pass data to each other. This is done using `output_key` and state variable injection:

**How it works:**
1. **Agent 1** generates output → stores in state with `output_key`
2. **Agent 2** references that output in its instruction using `{output_key}`
3. **Agent 3** can reference outputs from both previous agents

**Example with State Passing:**

```json
// Step 1: Writer generates code
{
  "agent": {
    "name": "code_writer_agent",
    "instruction": "Write clean code based on requirements. Output ONLY the code.",
    "output_key": "generated_code"
  }
}

// Step 2: Reviewer reads generated_code from state
{
  "agent": {
    "name": "code_reviewer_agent",
    "instruction": "Review this code:\n```python\n{generated_code}\n```\nProvide feedback.",
    "output_key": "review_comments"
  }
}

// Step 3: Refactorer reads both generated_code and review_comments
{
  "agent": {
    "name": "code_refactorer_agent",
    "instruction": "Original Code:\n{generated_code}\nReview:\n{review_comments}\nRefactor the code.",
    "output_key": "refactored_code"
  }
}
```

**Key Points:**
- Use `{variable_name}` in instructions to inject state variables
- Each agent's `output_key` becomes available to subsequent agents
- State is shared across all agents in the pipeline
- Variables are automatically injected before the agent processes the request

### Configuration Fields for Sequential Agents

- **agent_type**: Must be set to `"sequential"`
- **agent.name**: Name of the pipeline
- **agent.description**: What the pipeline does
- **sub_agents**: Array of sub-agent names (required, runs in order)
- **agent.model**: Not used (sub-agents have their own models)
- **agent.instruction**: Not needed (it's just orchestration)
- **tools**: Typically not used (sub-agents can have tools)

### Configuration Fields for Sub-Agents in Sequential Pipelines

- **agent.name**: Name of the agent step
- **agent.model**: Model to use for this step
- **agent.description**: What this step does
- **agent.instruction**: Instructions with `{state_variable}` placeholders for dynamic content
- **agent.output_key**: (Optional) Key name to store this agent's output in shared state
- **tools**: (Optional) Tools this agent can use

## Structured Output (JSON Schemas)

For production applications, you often need agents to return data in a consistent, parseable format rather than free-form text. The agent template supports **structured JSON output** through response schemas.

### Why Use Structured Output?

- ✅ **Consistent Format**: Always get data in the expected structure
- ✅ **Type Safety**: Define types for each field (string, number, array, etc.)
- ✅ **Easy Integration**: Directly use response data in your application without parsing
- ✅ **Validation**: Ensure required fields are present
- ✅ **Documentation**: Schema serves as API documentation

### Configuration

Add a `response_schema` field to your agent configuration:

```json
{
  "app_name": "product_analyzer",
  "agent_type": "llm",
  "agent": {
    "name": "product_analyzer",
    "model": "gemini-2.0-flash-exp",
    "description": "Analyzes products and returns structured data",
    "instruction": "Extract key information from product descriptions.",
    "response_schema": {
      "type": "object",
      "properties": {
        "product_name": {
          "type": "string",
          "description": "Name of the product"
        },
        "category": {
          "type": "string",
          "description": "Product category"
        },
        "key_features": {
          "type": "array",
          "items": { "type": "string" },
          "description": "List of key features"
        },
        "price_range": {
          "type": "string",
          "description": "Price range (budget/mid-range/premium)"
        }
      },
      "required": ["product_name", "category"]
    }
  }
}
```

### Usage in Code

When an agent has a `response_schema`, responses are automatically parsed as JSON:

```python
# Agent with response_schema automatically returns dict
agent = AgentManager("configs/product_analyzer_agent.json")
await agent.initialize()

response = await agent.send_message("Analyze: iPhone 15 Pro Max...")

# Response is a Python dict, not a string!
print(response["product_name"])  # "iPhone 15 Pro Max"
print(response["category"])      # "Electronics"
print(response["key_features"])  # ["A17 Pro chip", "Titanium design", ...]

# You can also explicitly request JSON parsing
response = await agent.send_message("...", return_json=True)
```

### UI Display

The Streamlit UI automatically detects structured output and displays it beautifully:

- 📊 **JSON viewer**: Collapsible, syntax-highlighted JSON display
- 📋 **Schema viewer**: View the response schema in agent details
- ✅ **Status indicator**: Shows "Structured JSON Output" badge
- 📥 **JSON downloads**: Download responses as `.json` files

### Schema Fields

The `response_schema` follows JSON Schema format:

- **type**: Data type (object, string, number, boolean, array)
- **properties**: Fields in an object
- **items**: Type of array elements
- **required**: Array of required field names
- **enum**: Allowed values for a field
- **description**: Field documentation (helps the AI understand)

### Example: Task Breakdown Agent

```json
{
  "response_schema": {
    "type": "object",
    "properties": {
      "project_title": { "type": "string" },
      "tasks": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "task_id": { "type": "integer" },
            "title": { "type": "string" },
            "priority": {
              "type": "string",
              "enum": ["high", "medium", "low"]
            },
            "estimated_time": { "type": "string" }
          },
          "required": ["task_id", "title", "priority"]
        }
      }
    },
    "required": ["project_title", "tasks"]
  }
}
```

**Usage:**

```python
agent = AgentManager("configs/task_breakdown_agent.json")
await agent.initialize()

response = await agent.send_message("Build a mobile app for fitness tracking")

# Access structured data
print(f"Project: {response['project_title']}")
for task in response['tasks']:
    print(f"  [{task['priority']}] {task['title']} - {task['estimated_time']}")
```

### Example Agents

The template includes example agents with structured output:

- **`product_analyzer_agent`**: Extracts product information
- **`task_breakdown_agent`**: Breaks down projects into tasks

Run examples:
```bash
python example_structured_output.py
```

### Tips

1. **Clear descriptions**: Help the AI by providing good field descriptions
2. **Required fields**: Mark essential fields as required
3. **Enums for categories**: Use enum for fields with limited options
4. **Nested objects**: You can nest objects and arrays for complex structures
5. **Model compatibility**: Works best with `gemini-2.0-flash-exp` or newer models

## Creating a New Agent

1. Create a new JSON configuration file in `configs/`:

```json
{
  "app_name": "my_custom_agent",
  "agent_type": "llm",
  "agent": {
    "name": "my_custom_agent",
    "model": "gemini-2.5-flash",
    "description": "A specialized agent for specific tasks.",
    "instruction": "You are an expert in [domain]. Help users with [specific tasks]."
  },
  "tools": []
}
```

2. Use the agent in your code:

```python
agent = AgentManager("configs/my_custom_agent.json")
await agent.initialize()
response = await agent.send_message("Your query here")
```

## Multiple Agents

You can run multiple agents simultaneously, each with its own configuration and session:

```python
agent1 = AgentManager("configs/template_agent.json")
agent2 = AgentManager("configs/code_reviewer_agent.json")

await agent1.initialize()
await agent2.initialize()

response1 = await agent1.send_message("Create a new agent")
response2 = await agent2.send_message("Review this code...")
```

## Web UI

ADK Agent Studio includes an interactive web interface built with Streamlit for easy testing and visualization.

### Features

- **Agent Selection**: Browse and select from all available agent configurations
- **Metadata Display**: View agent details including:
  - Agent type (LLM or Sequential)
  - Model information
  - Tools and sub-agents
  - Instructions and descriptions
  - Output keys (for sequential workflows)
  - Structured output schemas
- **Chat Mode**: Maintain conversation context across multiple messages
- **Interactive Execution**: Test agents with custom queries
- **Real-Time Results**: See agent responses instantly (text or structured JSON)
- **Execution Details**: View session IDs, user IDs, and execution metadata
- **Example Queries**: Pre-filled examples for each agent type
- **Download Results**: Save agent responses to file
- **Agent Creation**: Build new agent configurations with an intuitive form
  - Interactive schema builder for structured output
  - Real-time validation and filename preview
  - Automatic space-to-underscore conversion for valid identifiers
- **AI-Powered Prompt Generation**: Auto-generate agent instructions using AI
  - Uses dedicated helper agent (`helper_prompt_generator_agent`)
  - Generates instructions based on agent name and description
  - Instant form population with generated content
- **Code Export**: Generate standalone Python code for any agent to use without the AgentManager
  - Production-ready code with all dependencies
  - Proper async/await implementation
  - Environment variable handling

### Using the Web UI

1. **Start the UI:**
```bash
# Windows
run_ui.bat

# Unix/Mac  
bash run_ui.sh

# Or use streamlit directly
streamlit run app.py
```

2. **Select an Agent:**
   - Use the dropdown in the sidebar to choose an agent
   - View its metadata in the "Agent Details" tab

3. **Chat with an Agent:**
   - Use the "Chat Mode" tab for interactive conversations
   - Click "Start Chat Session" to begin
   - Chat history is maintained throughout the session
   - Type your messages in the chat input box
   - View conversation history in the chat interface
   - Use "New Chat" to reset and start fresh
   - Use "Clear History" to remove all messages
   - "Check ADK Session History" button for debugging conversation state

4. **Execute Single Queries:**
   - Switch to the "Single Execution" tab
   - Enter your query in the text area
   - Click "Execute" to run
   - View the response (text or structured JSON)
   - Download results as needed

5. **Create New Agents:**
   - Navigate to the "Create Agent" tab (far right)
   - **Enable Structured Output** (optional): Check the box before filling the form to define JSON schemas
   - Fill in the form with agent details:
     - **App Name**: Unique identifier (spaces auto-converted to underscores)
     - **Agent Name**: Display name (spaces auto-converted to underscores)
     - **Agent Type**: Choose LLM (intelligent) or Sequential (workflow)
     - **Model**: Select Gemini 2.5 Flash or Pro
     - **Description**: What the agent does
     - **Instructions**: System prompt defining behavior
   - **AI Prompt Generation**: 
     - Fill in Agent Name and Description
     - Click "✨ Generate with AI" below the Instructions field
     - AI generates comprehensive instructions automatically
     - Instructions populate the form immediately
   - **Structured Output Schema Builder** (if enabled):
     - Define number of output fields
     - For each field specify: name, type, description, required status
     - Support for strings, integers, numbers, booleans, arrays
     - Add enums for restricted string values
   - **Select Tools and Sub-Agents** as needed
   - **Preview** or **Save** the configuration
   - Click "Reload UI" to see your new agent in the selector

6. **Export Agent Code:**
   - Go to the "Agent Details" tab
   - Click "📤 Generate Code"
   - View the generated standalone Python code
   - Download the code as a `.py` file
   - Use the code in other projects without the AgentManager
   - The exported code includes:
     - All agent configuration (model, instructions, etc.)
     - Correct imports (google-adk, Runner, types, dotenv)
     - Environment variable loading (for API keys)
     - Tool imports with proper snake_case filenames
     - Class-based tool instantiation and execute method extraction
     - Structured output schemas as Pydantic models (if defined)
     - Complete usage example with async/await
     - Proper session and runner setup
     - Ready-to-run implementation
   - Prerequisites listed in code comments:
     - GOOGLE_API_KEY in .env file
     - Required Python packages
     - Tool files (if using tools)

7. **Tips:**
   - Check the "Example Queries" in the sidebar for ideas
   - Use the "Agent Details" tab to understand what the agent can do
   - Download responses for later reference
   - View raw configuration to see the full agent setup
   - Use AI generation to quickly create high-quality agent instructions
   - Export agents as code to use them in production applications

### UI Layout

The web interface features a modern, intuitive design with:

**Sidebar:**
- Agent selector dropdown
- Example queries for the selected agent
- Visual feedback on agent type and features

**Main Tabs:**
- **💬 Chat Mode**: Interactive conversation interface with persistent history
- **🎯 Single Execution**: One-time query testing with immediate results
- **📊 Agent Details**: View configuration, metadata, and export code
- **➕ Create Agent**: Form-based agent creation with AI assistance (separated on far right)

**Agent Details Tab Features:**
- Configuration overview with info icon
- Agent type, model, and capabilities
- Tools and sub-agents list
- Instructions and description
- Structured output schema viewer (if configured)
- Code export section with download
- Raw JSON configuration expander

**Create Agent Tab Features:**
- Structured form with validation
- Real-time filename preview
- AI prompt generation button
- Interactive schema builder
- Tool and sub-agent selection
- Preview and save options
- Auto-reload functionality

## Project Structure

```
adk-agent-studio/
├── agent_def/
│   ├── __init__.py           # Exports AgentManager
│   ├── agent.py              # AgentManager class implementation
│   ├── base_tool.py          # BaseTool abstract class
│   └── tool_loader.py        # Tool loading system
├── configs/
│   ├── 01_basic_template_agent.json      # Basic LLM agent
│   ├── 02_tools_assistant_agent.json     # Agent with calculator & web search
│   ├── 03_subagents_coordinator_agent.json # Coordinator with sub-agents
│   ├── 04_subagents_greeting_agent.json  # Greeting specialist
│   ├── 05_subagents_farewell_agent.json  # Farewell specialist
│   ├── 06_sequential_code_pipeline_agent.json # Sequential workflow
│   ├── 07_sequential_code_writer_agent.json   # Code writing step
│   ├── 08_sequential_code_reviewer_agent.json # Code review step
│   ├── 09_sequential_code_refactorer_agent.json # Code refactor step
│   ├── 10_structured_product_analyzer_agent.json # JSON output example
│   ├── 11_structured_task_breakdown_agent.json   # Task breakdown with schema
│   ├── helper_prompt_generator_agent.json # AI prompt generation helper
│   └── ...                              # Your custom agents
├── tools/
│   ├── calculator.py         # Function-based tool
│   ├── web_search_tool.py    # Class-based tool
│   └── get_weather.py        # Weather tool
├── app.py                    # Streamlit Web UI
├── run_ui.bat                # Windows UI launcher
├── run_ui.sh                 # Unix/Mac UI launcher
├── example_usage.py          # Code examples (basic usage)
├── example_structured_output.py # Structured output examples
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (create with GOOGLE_API_KEY)
└── README.md                 # This file
```

## API Reference

### AgentManager

The main class for managing agent lifecycle.

#### Methods

**`__init__(config_path: str)`**
- Initialize an agent manager with a configuration file
- Generates unique user_id and session_id
- Creates session service

**`async initialize()`**
- Create the agent from config
- Set up the session
- Create the runner
- Must be called before sending messages

**`async send_message(query: str) -> str`**
- Send a message to the agent
- Returns the agent's response as a string

**`async close()`**
- Cleanup resources (placeholder for future use)

## Advanced Usage

### Multi-turn Conversations

Agents maintain conversation history within the same session:

```python
agent = AgentManager("configs/template_agent.json")
await agent.initialize()

# First message
response1 = await agent.send_message("I need help with project planning")

# Follow-up message (agent remembers context)
response2 = await agent.send_message("What about timeline estimation?")

await agent.close()
```

### Environment Variables

You can use environment variables in your code by loading them with `python-dotenv`:

```python
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("API_KEY")
model_name = os.getenv("MODEL_NAME", "gemini-2.5-flash")
```

## Troubleshooting

### Module not found errors

Make sure you've activated the virtual environment:

```bash
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Windows (CMD)
.\venv\Scripts\activate.bat

# Unix/MacOS
source venv/bin/activate
```

## License

MIT License - feel free to use this template for your own projects!

