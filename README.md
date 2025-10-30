# ADK Agent Manager

> A comprehensive development environment for Google ADK (Agent Development Kit)

Build, test, and deploy production-ready AI agents with an interactive UI, config-based creation, and zero boilerplate. Features include hierarchical agent structures, tool integration, sequential workflows, structured output, MCP server management, and instant code export.

## Features

- **Interactive Web UI**: Streamlit interface for creating, testing, and managing agents
- **Configuration-driven**: Define agents using JSON files with zero boilerplate
- **Hierarchical Agents**: Sub-agents and coordinator patterns for complex workflows
- **Sequential Pipelines**: Multi-step workflows with state passing between agents
- **Tool System**: Function-based and class-based tools with auto-discovery
- **MCP Integration**: Create and manage Model Context Protocol servers, connect agents to MCP tools
- **Structured Output**: JSON schemas with Pydantic validation for consistent responses
- **AI-Powered**: Auto-generate agent instructions and custom tools using AI
- **Code Export**: Generate standalone Python code from any agent configuration
- **Session Management**: Automatic conversation history and context retention

## Installation

### 1. Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### 2. Clone and Setup

```bash
git clone <your-repository-url>
cd adk-agent-manager

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Unix/Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables

Create `agent_def/.env` from the template:

```bash
cd agent_def
cp .env.example .env  # or create manually
```

Edit `agent_def/.env` with your configuration:

```env
# Google Cloud / Vertex AI Configuration
GOOGLE_GENAI_USE_VERTEXAI=false
GOOGLE_CLOUD_PROJECT=your_project_id
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_APPLICATION_CREDENTIALS=agent_def/sa/your-service-account-key.json

# Optional: Tool APIs
WEATHER_API_KEY=your_weather_api_key_here
SEARCH_API_KEY=your_search_api_key_here
SEARCH_ENGINE=google

# Optional: MCP Server Configuration
PORT=8000
TRANSPORT=http
```

**Getting API Keys:**
- **Google AI API Key**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey) (required if not using Vertex AI)
- **Vertex AI**: Set `GOOGLE_GENAI_USE_VERTEXAI=true` and provide project ID, location, and service account credentials
- **Weather/Search APIs**: Optional, for enhanced tool functionality

**Note:** The `.env` file is git-ignored. Never commit API keys or credentials to version control.

## Quick Start

### Web UI (Recommended)

Launch the interactive interface:

```bash
streamlit run app.py
# Or use the platform-specific scripts:
# Windows: run_ui.bat
# Unix/Mac: bash run_ui.sh
```

Open `http://localhost:8501` to:
- Browse and test pre-configured agents
- Chat with agents (conversation history maintained)
- Create new agents with AI-assisted form
- Build MCP servers from your tools
- Export agents as standalone Python code

### Python Code

```python
import asyncio
from agent_def import AgentManager

async def main():
    # Create agent from config
    agent = AgentManager("configs/01_basic_template_agent.json")
    await agent.initialize()
    
    # Send messages
    response = await agent.send_message("Hello! How can you help me?")
    print(response)
    
    # Cleanup
    await agent.close()

asyncio.run(main())
```

### Example Scripts

```bash
python example_usage.py                # Basic agent usage
python example_structured_output.py    # JSON schema examples
python example_mcp_usage.py            # MCP server examples
python example_mcp_integration.py      # Agent + MCP integration
```

## Core Concepts

### Agent Configuration

Agents are defined in JSON files stored in `configs/`. Basic structure:

```json
{
  "app_name": "my_agent",
  "agent_type": "llm",
  "agent": {
    "name": "my_agent",
    "model": "gemini-2.5-flash",
    "description": "Brief description of what this agent does",
    "instruction": "You are an expert assistant. Help users with..."
  },
  "tools": ["calculator", "WebSearchTool"],
  "sub_agents": ["specialist_agent"],
  "mcp_servers": [{"config": "calculator_server"}],
  "example_queries": ["What can you help me with?"]
}
```

**Key Fields:**
- **app_name**: Unique identifier for the agent
- **agent_type**: `"llm"` (intelligent) or `"sequential"` (workflow pipeline)
- **agent.name**: Display name (must be valid Python identifier)
- **agent.model**: AI model (e.g., "gemini-2.5-flash", "gemini-2.5-pro")
- **agent.instruction**: System prompt defining behavior (required for LLM agents)
- **agent.response_schema**: JSON schema for structured output (optional)
- **agent.output_key**: State variable name for sequential workflows (optional)
- **tools**: Array of tool names from `tools/` directory (optional)
- **sub_agents**: Array of sub-agent config names (optional)
- **mcp_servers**: Array of MCP server connections (optional)
- **example_queries**: Array of sample queries for UI display (optional)

### Tools

Tools extend agent capabilities. Two types are supported:

**Function-based tools** (simple, stateless):
```python
# tools/my_tool.py
async def my_tool(param: str) -> str:
    """Tool description."""
    return f"Processed: {param}"
```

**Class-based tools** (with state, setup/teardown):
```python
# tools/my_advanced_tool.py
from agent_def.base_tool import BaseTool, ToolParameter

class MyAdvancedTool(BaseTool):
    @property
    def name(self) -> str:
        return "my_advanced_tool"
    
    @property
    def description(self) -> str:
        return "What this tool does"
    
    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="input", type="string", required=True)
        ]
    
    async def execute(self, **kwargs) -> str:
        params = self.validate_parameters(**kwargs)
        return f"Result: {params['input']}"
```

**Using Tools:**
Add tool names to agent config's `tools` array. Configure via environment variables in `.env`:

```env
SEARCH_API_KEY=your_api_key
DATABASE_URL=postgresql://localhost/db
```

### Sub-Agents

Create hierarchical agent systems where coordinators delegate to specialists:

```json
{
  "agent": {
    "name": "coordinator",
    "instruction": "Delegate greetings to greeting_agent, farewells to farewell_agent"
  },
  "tools": ["get_weather"],
  "sub_agents": ["greeting_agent", "farewell_agent"]
}
```

Sub-agents:
- Share the parent's session (conversation history)
- Can have their own tools and sub-agents (nested hierarchies)
- Are loaded automatically from `configs/{agent_name}.json`

### Sequential Workflows

For fixed pipelines where agents run in order:

```json
{
  "agent_type": "sequential",
  "agent": {
    "name": "CodePipelineAgent",
    "description": "Write → Review → Refactor pipeline"
  },
  "sub_agents": [
    "code_writer_agent",
    "code_reviewer_agent",
    "code_refactorer_agent"
  ]
}
```

**State Passing:** Use `output_key` to pass data between pipeline steps:

```json
// Step 1: Writer
{
  "agent": {
    "name": "code_writer_agent",
    "instruction": "Write code based on requirements",
    "output_key": "generated_code"
  }
}

// Step 2: Reviewer (uses {generated_code})
{
  "agent": {
    "name": "code_reviewer_agent",
    "instruction": "Review:\n```\n{generated_code}\n```\nProvide feedback",
    "output_key": "review_comments"
  }
}

// Step 3: Refactorer (uses {generated_code} and {review_comments})
{
  "agent": {
    "name": "code_refactorer_agent",
    "instruction": "Original: {generated_code}\nReview: {review_comments}\nRefactor it",
    "output_key": "refactored_code"
  }
}
```

### Structured Output

Define JSON schemas for consistent, parseable responses:

```json
{
  "agent": {
    "name": "product_analyzer",
    "instruction": "Extract product information",
    "response_schema": {
      "type": "object",
      "properties": {
        "product_name": {
          "type": "string",
          "description": "Product name"
        },
        "category": {
          "type": "string",
          "description": "Product category"
        },
        "key_features": {
          "type": "array",
          "items": {"type": "string"},
          "description": "List of features"
        },
        "price_range": {
          "type": "string",
          "enum": ["budget", "mid-range", "premium"]
        }
      },
      "required": ["product_name", "category"]
    }
  }
}
```

**Usage:**
```python
agent = AgentManager("configs/10_structured_product_analyzer_agent.json")
await agent.initialize()

response = await agent.send_message("Analyze: iPhone 15 Pro...")

# Response is a Python dict, not a string
print(response["product_name"])      # "iPhone 15 Pro"
print(response["category"])          # "Electronics"
print(response["key_features"])      # ["A17 Pro chip", ...]
```

### MCP Integration

Connect agents to Model Context Protocol servers for distributed tool access.

**MCP Server Configuration** (`configs_mcp/calculator_server.json`):
```json
{
  "server": {
    "name": "Calculator Server",
    "description": "MCP server with calculator tools",
    "transport": "http",
    "host": "0.0.0.0",
    "port": 8000
  },
  "tools": [
    {
      "file": "calculator.py",
      "type": "function",
      "function_name": "calculator"
    }
  ]
}
```

**Transport Options:**
- **stdio**: For Claude Desktop / IDE integration
- **http**: For network access
- **sse**: For streaming updates

**Using MCP Servers in Agents:**
```json
{
  "agent": {
    "name": "mcp_agent",
    "instruction": "You have access to calculator tools"
  },
  "mcp_servers": [
    {
      "config": "01_calculator_server",
      "url": "http://localhost:8000",
      "auto_start": false
    }
  ]
}
```

**MCP Server Options:**
- **config**: Reference to config file in `configs_mcp/` (without `.json`)
- **url**: Direct URL to connect to (optional if in config file)
- **auto_start**: Start server automatically when agent initializes (requires http/sse)

**Connection Scenarios:**
1. **Connect to running server**: Provide `url` only
2. **Auto-start server**: Set `auto_start: true` with `config` (http/sse only)
3. **Mixed tools**: Combine local `tools` array with `mcp_servers`

**Example:**
```python
# Agent auto-starts MCP server and uses its tools
agent = AgentManager("configs/14_mcp_auto_start_agent.json")
await agent.initialize()  # Server starts in background

response = await agent.send_message("Calculate 123 * 456")

await agent.close()  # Server stops automatically
```

## Web UI Guide

The Streamlit interface provides a complete environment for agent development.

### Main Tabs

**💬 Chat Mode**
- Interactive conversations with agents
- Persistent conversation history
- Multi-turn context retention
- "New Chat" and "Clear History" controls
- Session state debugging tools

**🎯 Single Execution**
- One-time query testing
- Immediate results (text or JSON)
- Download responses
- Example queries sidebar

**📊 Agent Details**
- Configuration overview
- Agent type, model, capabilities
- Tools, sub-agents, and MCP servers list
- Instructions and descriptions
- Structured output schema viewer
- Code export with download
- Raw JSON configuration

**🔌 MCP Servers**
- Browse available MCP servers
- View server configurations and tools
- Initialize and test servers
- Export config for Claude Desktop/Cursor
- Usage instructions

**➕ Create Agent**
- Interactive form for new agents
- AI-powered instruction generation
- Tool and sub-agent selection
- MCP server integration
- Structured output schema builder
- Real-time validation
- Preview and save

**➕ Create MCP Server**
- Build MCP servers from existing tools
- Transport protocol selection (stdio/http/sse)
- Tool selection with type detection
- Configuration preview
- Save and reload

### Creating Agents via UI

1. Navigate to **➕ Create Agent** tab
2. Fill in basic information:
   - App Name (unique identifier)
   - Agent Type (LLM or Sequential)
   - Model (Gemini 2.5 Flash/Pro)
   - Description
3. **Generate Instructions with AI** (optional):
   - Enter name and description
   - Click **✨ Generate with AI**
   - AI populates the instructions field
4. **Add Tools & Sub-Agents** (optional):
   - Select from `tools/` directory
   - Choose sub-agents from `configs/`
5. **Connect MCP Servers** (optional):
   - Select servers from `configs_mcp/`
   - Configure URL and auto-start options
6. **Add Example Queries** (optional):
   - Define sample queries for testing
7. **Enable Structured Output** (optional):
   - Toggle schema builder
   - Define output fields with types
8. **Preview** or **Create** agent
9. Click **Reload UI** to see new agent

### Creating MCP Servers via UI

1. Navigate to **➕ Create MCP Server** tab
2. **Write Custom Function** (optional):
   - Expand "Create New Function"
   - Enter function name and description
   - Click **✨ Generate with AI** for automatic code
   - Edit generated code as needed
   - Click **💾 Save Function**
   - Click **🔄 Refresh Tools List**
3. Configure server:
   - Server Name
   - Config File Name
   - Description
   - Transport protocol (stdio/http/sse)
   - Host and Port (for http/sse)
4. Select tools to include
5. **Preview Config** or **Create Server**

### Exporting Agent Code

1. Select an agent
2. Go to **📊 Agent Details** tab
3. Click **📤 Generate Code**
4. View standalone Python code
5. Download as `.py` file

The exported code includes:
- All imports and dependencies
- Tool imports with proper instantiation
- Pydantic models for structured output
- MCP toolset initialization (if used)
- Session and runner setup
- Complete usage example
- Prerequisites in comments

## Project Structure

```
adk-agent-manager/
├── agent_def/
│   ├── __init__.py              # Exports AgentManager
│   ├── agent.py                 # AgentManager implementation
│   ├── base_tool.py             # BaseTool abstract class
│   ├── tool_loader.py           # Tool loading system
│   ├── .env                     # Environment variables (create this)
│   └── .env.example             # Environment template
├── mcp_def/
│   ├── __init__.py              # MCP package init
│   ├── mcp_manager.py           # MCPServerManager class
│   └── server.py                # Standalone MCP server template
├── configs/                     # Agent configurations
│   ├── 01_basic_template_agent.json
│   ├── 02_tools_assistant_agent.json
│   ├── 03_subagents_coordinator_agent.json
│   ├── 10_structured_product_analyzer_agent.json
│   ├── 12_mcp_calculator_agent.json
│   ├── helper_prompt_generator_agent.json
│   └── ...
├── configs_mcp/                 # MCP server configurations
│   ├── 01_calculator_server.json
│   ├── 02_weather_server.json
│   ├── 03_multi_tool_server.json
│   └── ...
├── tools/                       # Tool implementations
│   ├── calculator.py            # Function-based tool
│   ├── web_search_tool.py       # Class-based tool
│   ├── get_weather.py           # Weather tool
│   └── ...
├── app.py                       # Streamlit Web UI
├── example_usage.py             # Basic usage examples
├── example_structured_output.py # Structured output examples
├── example_mcp_usage.py         # MCP server examples
├── example_mcp_integration.py   # Agent + MCP examples
├── requirements.txt             # Python dependencies
├── run_ui.bat                   # Windows UI launcher
├── run_ui.sh                    # Unix/Mac UI launcher
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## API Reference

### AgentManager

**`AgentManager(config_path: str)`**
- Initialize agent manager with configuration file
- Automatically generates unique user_id and session_id

**`async initialize()`**
- Create agent from config
- Load tools and sub-agents
- Connect to MCP servers
- Set up session and runner
- Must be called before sending messages

**`async send_message(query: str, return_json: bool = False) -> str | dict`**
- Send message to agent
- Returns string by default
- Returns dict if `return_json=True` or agent has `response_schema`

**`get_conversation_history() -> list[dict]`**
- Get conversation history from session
- Returns list of message dicts with `role` and `content`

**`async close()`**
- Cleanup resources
- Stop auto-started MCP servers
- Close agent connections

### MCPServerManager

**`MCPServerManager(config_path: str)`**
- Initialize MCP server manager with config file

**`initialize()`**
- Load and register all tools from config
- Prepare server for start

**`start_server()`**
- Start MCP server (blocks for stdio, background for http/sse)

**`stop_server()`**
- Stop running server (for http/sse)

**`get_config() -> dict`**
- Get server configuration

**`export_config_for_claude() -> dict`**
- Export Claude Desktop compatible config

## Advanced Usage

### Multi-turn Conversations

Agents maintain conversation history within sessions:

```python
agent = AgentManager("configs/01_basic_template_agent.json")
await agent.initialize()

# Context is preserved across messages
await agent.send_message("My name is Alice")
await agent.send_message("What's my name?")  # Agent remembers "Alice"

# Get full history
history = agent.get_conversation_history()
for msg in history:
    print(f"{msg['role']}: {msg['content']}")
```

### Environment Variables

Load configuration from `.env` using `python-dotenv`:

```python
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
project = os.getenv("GOOGLE_CLOUD_PROJECT")
```

### Claude Desktop Integration

To use MCP servers with Claude Desktop:

1. Create MCP server in UI or via config
2. In **MCP Servers** tab, select server
3. Click **📥 Export for Claude**
4. Copy JSON configuration
5. Add to Claude Desktop config:
   - **Mac**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "Calculator Server": {
      "command": "python",
      "args": ["-m", "mcp_def.mcp_manager", "configs_mcp/01_calculator_server.json"],
      "description": "Calculator tools for Claude"
    }
  }
}
```

## Troubleshooting

**"Module not found" errors:**
```bash
# Ensure virtual environment is activated
# Windows:
venv\Scripts\activate
# Unix/Mac:
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

**"GOOGLE_API_KEY not found":**
- Create `agent_def/.env` file
- Add `GOOGLE_API_KEY=your_key_here`
- Restart application

**"Failed to connect to MCP server":**
- Ensure MCP server is running at specified URL
- Check firewall settings
- Verify port is not in use
- For auto-start: ensure http/sse transport (not stdio)

**"Agent name validation error":**
- Agent names must be valid Python identifiers
- Use underscores instead of spaces
- Start with letter or underscore
- UI automatically converts spaces to underscores

**"Tools not loading":**
- Check tool files exist in `tools/` directory
- Verify function/class names match config
- For class tools, check inheritance from `BaseTool`
- Review tool logs in console

## License

MIT License - feel free to use this template for your own projects!
