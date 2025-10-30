"""
Streamlit Frontend for ADK Agent Manager
Interactive UI for testing and visualizing agent execution.
"""
import streamlit as st
import asyncio
import json
import re
from pathlib import Path
from datetime import datetime
from agent_def import AgentManager
from mcp_def.mcp_manager import MCPServerManager

def generate_agent_code(config: dict) -> str:
    """
    Generate standalone Python code from an agent configuration.
    """
    lines = []
    
    # Header comment
    lines.append("\"\"\"")
    lines.append(f"Standalone Agent Code - {config.get('app_name', 'agent')}")
    lines.append("Generated from agent configuration")
    lines.append("")
    lines.append("Prerequisites:")
    lines.append("- Create a .env file with your GOOGLE_API_KEY (required)")
    lines.append("- Ensure all tool files are available in the 'tools/' directory (if using tools)")
    lines.append("- Install required packages: google-adk, pydantic, python-dotenv")
    lines.append("")
    lines.append("Example .env file:")
    lines.append("  GOOGLE_API_KEY=your_api_key_here")
    lines.append("\"\"\"")
    lines.append("")
    
    # Imports
    lines.append("import asyncio")
    
    # Get config values
    agent_type = config.get("agent_type", "llm")
    agent_config = config.get("agent", {})
    tools = config.get("tools", [])
    sub_agents = config.get("sub_agents", [])
    mcp_servers = config.get("mcp_servers", [])
    
    if agent_type == "sequential":
        lines.append("from google.adk.agents import SequentialAgent")
    else:
        lines.append("from google.adk.agents import Agent")
    
    lines.append("from google.adk.sessions import Session, InMemorySessionService")
    lines.append("from google.adk.runners import Runner")
    lines.append("from google.genai import types")
    lines.append("from dotenv import load_dotenv")
    lines.append("import uuid")
    
    # Add MCP toolset import if needed
    if mcp_servers:
        lines.append("from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset")
    
    lines.append("")
    
    # Load environment variables (required for Google AI API key)
    lines.append("# Load environment variables (required for API keys)")
    lines.append("load_dotenv()")
    lines.append("")
    
    # Add structured output import if needed
    if "response_schema" in agent_config:
        lines.append("from pydantic import BaseModel, Field")
    
    lines.append("")
    
    # Tool imports
    if tools:
        lines.append("# Tool imports")
        for tool in tools:
            # Convert tool name to snake_case for file name
            # e.g., WebSearchTool -> web_search_tool
            file_name = re.sub(r'(?<!^)(?=[A-Z])', '_', tool).lower()
            lines.append(f"from tools.{file_name} import {tool}")
        lines.append("")
        
        # Check which tools are classes and need instantiation
        lines.append("# Initialize class-based tools")
        lines.append("# Note: Class-based tools (inheriting from BaseTool) need to be instantiated")
        class_tools = []
        for tool in tools:
            # Heuristic: if tool name is PascalCase, it's likely a class
            if tool[0].isupper() and any(c.isupper() for c in tool[1:]):
                class_tools.append(tool)
                lines.append(f"{tool.lower()}_instance = {tool}()")
        
        if class_tools:
            lines.append("")
            lines.append("# For Google ADK compatibility, extract execute methods from class tools")
            for tool in class_tools:
                lines.append(f"{tool.lower()}_tool = {tool.lower()}_instance.execute")
        lines.append("")
    
    # Structured output schema definition
    if "response_schema" in agent_config:
        lines.append("# Define output schema")
        schema = agent_config["response_schema"]
        model_name = f"{agent_config.get('name', 'Agent').replace('_', '')}Output"
        lines.append(f"class {model_name}(BaseModel):")
        
        properties = schema.get("properties", {})
        required_fields = schema.get("required", [])
        
        for field_name, field_info in properties.items():
            field_type = field_info.get("type", "string")
            field_desc = field_info.get("description", "")
            is_required = field_name in required_fields
            
            # Map JSON types to Python types
            type_map = {
                "string": "str",
                "integer": "int",
                "number": "float",
                "boolean": "bool",
                "array": "list"
            }
            python_type = type_map.get(field_type, "str")
            
            # Add enum info if present
            enum_values = field_info.get("enum")
            enum_comment = f"  # Options: {enum_values}" if enum_values else ""
            
            if is_required:
                lines.append(f"    {field_name}: {python_type} = Field(description=\"{field_desc}\"){enum_comment}")
            else:
                lines.append(f"    {field_name}: {python_type} | None = Field(default=None, description=\"{field_desc}\"){enum_comment}")
        
        lines.append("")
    
    # Sub-agent definitions (if any)
    if sub_agents:
        lines.append("# Sub-agent definitions would need to be loaded here")
        lines.append("# For this example, assume they're already defined or imported")
        lines.append("")
    
    # MCP server connections (if any)
    if mcp_servers:
        lines.append("# Connect to MCP servers")
        lines.append("mcp_toolsets = []")
        lines.append("")
        
        for idx, mcp_server in enumerate(mcp_servers):
            url = mcp_server.get("url")
            config_ref = mcp_server.get("config")
            
            # Generate comment about the server
            if config_ref:
                lines.append(f"# MCP server from config: {config_ref}")
            if url:
                lines.append(f"# Connecting to: {url}")
            
            # Determine URL to use
            if url:
                server_url = url
            elif config_ref:
                # Try to infer URL from config (user would need to adjust if different)
                server_url = f"http://localhost:800{idx}"
                lines.append(f"# Note: Update URL if different from default")
            else:
                continue
            
            lines.append(f"mcp_toolset_{idx} = MCPToolset(url=\"{server_url}\")")
            lines.append(f"mcp_toolsets.append(mcp_toolset_{idx})")
            lines.append("")
    
    # Main agent definition
    lines.append("# Define the agent")
    agent_name_var = f"{config.get('app_name', 'agent').replace('-', '_')}_agent"
    
    if agent_type == "sequential":
        lines.append(f"{agent_name_var} = SequentialAgent(")
    else:
        lines.append(f"{agent_name_var} = Agent(")
    
    # Agent parameters
    lines.append(f"    name=\"{agent_config.get('name', 'agent')}\",")
    
    if agent_type != "sequential":
        lines.append(f"    model=\"{agent_config.get('model', 'gemini-2.5-flash')}\",")
    
    description = agent_config.get('description', '').replace('"', '\\"')
    lines.append(f"    description=\"{description}\",")
    
    if agent_type != "sequential":
        instruction = agent_config.get('instruction', '').replace('"', '\\"').replace('\n', '\\n')
        lines.append(f"    instruction=\"{instruction}\",")
    
    # Tools
    if tools:
        # Build tool references - use execute method for classes, direct reference for functions
        tool_refs = []
        for tool in tools:
            # If it's a class (PascalCase), use the extracted execute method
            if tool[0].isupper() and any(c.isupper() for c in tool[1:]):
                tool_refs.append(f"{tool.lower()}_tool")
            else:
                # It's a function
                tool_refs.append(tool)
        tools_str = ", ".join(tool_refs)
        lines.append(f"    tools=[{tools_str}],")
    
    # Sub-agents
    if sub_agents:
        sub_agents_str = ", ".join([f"{sa}_agent" for sa in sub_agents])
        lines.append(f"    sub_agents=[{sub_agents_str}],")
    
    # Output key
    if "output_key" in agent_config:
        lines.append(f"    output_key=\"{agent_config['output_key']}\",")
    
    # Output schema
    if "response_schema" in agent_config:
        lines.append(f"    output_schema={model_name},")
    
    # MCP toolsets
    if mcp_servers:
        lines.append(f"    toolsets=mcp_toolsets,")
    
    lines.append(")")
    lines.append("")
    
    # Define root_agent for ADK compatibility
    lines.append("# For ADK tools compatibility, the root agent must be named 'root_agent'")
    lines.append(f"root_agent = {agent_name_var}")
    lines.append("")
    
    # Usage example
    lines.append("# Usage example")
    lines.append("async def main():")
    lines.append("    # Generate unique user ID")
    lines.append("    user_id = str(uuid.uuid4())")
    lines.append("")
    lines.append("    # Create session")
    lines.append("    session_service = InMemorySessionService()")
    lines.append(f"    session = await session_service.create_session(app_name=\"{config.get('app_name', 'agent')}\", user_id=user_id)")
    lines.append("")
    lines.append("    # Create runner")
    lines.append(f"    runner = Runner(")
    lines.append(f"        agent=root_agent,")
    lines.append(f"        app_name=\"{config.get('app_name', 'agent')}\",")
    lines.append(f"        session_service=session_service")
    lines.append(f"    )")
    lines.append("")
    lines.append("    # Run the agent")
    lines.append("    query = \"Your query here\"")
    lines.append("    print(f\"Query: {{query}}\")")
    lines.append("    print(\"\\nAgent Response:\")")
    lines.append("")
    lines.append("    # Create message content")
    lines.append("    content = types.Content(role=\"user\", parts=[types.Part(text=query)])")
    lines.append("")
    lines.append("    # Run the conversation")
    lines.append("    async for event in runner.run_async(")
    lines.append("        user_id=user_id,")
    lines.append("        session_id=session.id,")
    lines.append("        new_message=content")
    lines.append("    ):")
    lines.append("        if event.is_final_response():")
    lines.append("            print(event.content)")
    lines.append("")
    lines.append("if __name__ == \"__main__\":")
    lines.append("    asyncio.run(main())")
    
    return "\n".join(lines)

# Page configuration
st.set_page_config(
    page_title="ADK Agent Manager",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
    }
    .metadata-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .execution-box {
        background-color: #f3f4ff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)


def get_available_agents():
    """Get list of available agent configurations."""
    configs_dir = Path("configs")
    if not configs_dir.exists():
        return []
    
    agents = []
    for config_file in configs_dir.glob("*.json"):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                agents.append({
                    "name": config_file.stem,
                    "path": str(config_file),
                    "config": config
                })
        except Exception as e:
            st.warning(f"Could not load {config_file.name}: {e}")
    
    return sorted(agents, key=lambda x: x["name"])


def get_available_mcp_servers():
    """Get list of available MCP server configurations."""
    configs_dir = Path("configs_mcp")
    if not configs_dir.exists():
        return []
    
    servers = []
    for config_file in configs_dir.glob("*.json"):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                servers.append({
                    "name": config_file.stem,
                    "path": str(config_file),
                    "config": config
                })
        except Exception as e:
            st.warning(f"Could not load {config_file.name}: {e}")
    
    return sorted(servers, key=lambda x: x["name"])


def get_available_tools():
    """Get list of available tools with type detection."""
    tools_dir = Path("tools")
    if not tools_dir.exists():
        return []
    
    tools = []
    for tool_file in tools_dir.glob("*.py"):
        if tool_file.stem == "__init__":
            continue
        
        # Try to detect if it's a function or class-based tool
        try:
            with open(tool_file, 'r') as f:
                content = f.read()
                
                # Simple heuristic: check if it has "class" and "BaseTool" or if it's a function
                is_class = "class " in content and ("BaseTool" in content or "execute(" in content)
                
                if is_class:
                    # Extract class name
                    import re
                    match = re.search(r'class\s+(\w+)', content)
                    class_name = match.group(1) if match else tool_file.stem.title().replace('_', '')
                    tools.append({
                        "file": tool_file.name,
                        "name": tool_file.stem,
                        "type": "class",
                        "class_name": class_name
                    })
                else:
                    # It's likely a function
                    # Extract function name
                    match = re.search(r'async def\s+(\w+)', content)
                    func_name = match.group(1) if match else tool_file.stem
                    tools.append({
                        "file": tool_file.name,
                        "name": tool_file.stem,
                        "type": "function",
                        "function_name": func_name
                    })
        except Exception:
            # Default to function type if we can't determine
            tools.append({
                "file": tool_file.name,
                "name": tool_file.stem,
                "type": "function",
                "function_name": tool_file.stem
            })
    
    return sorted(tools, key=lambda x: x["name"])


def display_agent_metadata(agent_info):
    """Display agent metadata in a nice format."""
    config = agent_info["config"]
    
    st.markdown("### 📋 Agent Metadata")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**App Name:**")
        st.code(config.get("app_name", "N/A"))
        
        st.markdown("**Agent Type:**")
        agent_type = config.get("agent_type", "llm").upper()
        type_emoji = "🧠" if agent_type == "LLM" else "⚙️"
        st.code(f"{type_emoji} {agent_type}")
        
        if "agent" in config:
            st.markdown("**Model:**")
            st.code(config["agent"].get("model", "N/A"))
    
    with col2:
        st.markdown("**Agent Name:**")
        st.code(config.get("agent", {}).get("name", "N/A"))
        
        if "tools" in config and config["tools"]:
            st.markdown("**Tools:**")
            for tool in config["tools"]:
                st.code(f"🔧 {tool}")
        
        if "sub_agents" in config and config["sub_agents"]:
            st.markdown("**Sub-Agents:**")
            for sub_agent in config["sub_agents"]:
                st.code(f"👥 {sub_agent}")
        
        if "mcp_servers" in config and config["mcp_servers"]:
            st.markdown("**MCP Servers:**")
            for mcp_server in config["mcp_servers"]:
                server_name = mcp_server.get("config", "Unknown")
                server_url = mcp_server.get("url", "")
                auto_start = mcp_server.get("auto_start", False)
                
                server_info = f"🔌 {server_name}"
                if auto_start:
                    server_info += " (auto-start)"
                elif server_url:
                    server_info += f" ({server_url})"
                
                st.code(server_info)
    
    # Description
    if "agent" in config and "description" in config["agent"]:
        st.markdown("**Description:**")
        st.markdown(f"""
        <div style="color: #6b7280; font-size: 0.95rem; line-height: 1.5; margin-bottom: 1.5rem;">
            {config["agent"]["description"]}
        </div>
        """, unsafe_allow_html=True)
    
    # Instructions (collapsible)
    if "agent" in config and "instruction" in config["agent"]:
        with st.expander("📝 View Instructions"):
            st.text(config["agent"]["instruction"])
    
    # Output key (for sequential agents)
    if "agent" in config and "output_key" in config["agent"]:
        st.markdown("**Output Key:**")
        st.code(f"🔑 {config['agent']['output_key']}")
    
    # Response schema (structured output)
    if "agent" in config and "response_schema" in config["agent"]:
        st.markdown("**Structured Output:**")
        st.success("✅ Enabled - Returns JSON")
        with st.expander("📋 View Response Schema"):
            st.json(config["agent"]["response_schema"])


async def execute_agent(agent_path, query):
    """Execute an agent with the given query (one-shot execution)."""
    try:
        # Create agent manager
        agent_manager = AgentManager(agent_path)
        
        # Initialize
        await agent_manager.initialize()
        
        # Check if agent has structured output
        has_schema = "response_schema" in agent_manager.config.get("agent", {})
        
        # Execute query
        response = await agent_manager.send_message(query)
        
        # Cleanup
        await agent_manager.close()
        
        return {
            "success": True,
            "response": response,
            "is_json": isinstance(response, dict),
            "has_schema": has_schema,
            "user_id": agent_manager.user_id,
            "session_id": agent_manager.session_id
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


async def initialize_chat_agent(agent_path):
    """Initialize an agent for chat mode (persistent session)."""
    try:
        agent_manager = AgentManager(agent_path)
        await agent_manager.initialize()
        return {
            "success": True,
            "agent_manager": agent_manager
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


async def send_chat_message(agent_manager, query):
    """Send a message in chat mode."""
    try:
        response = await agent_manager.send_message(query)
        return {
            "success": True,
            "response": response,
            "is_json": isinstance(response, dict)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def main():
    """Main Streamlit app."""
    
    # Initialize session state
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_agent_manager" not in st.session_state:
        st.session_state.chat_agent_manager = None
    if "chat_agent_name" not in st.session_state:
        st.session_state.chat_agent_name = None
    if "chat_initialized" not in st.session_state:
        st.session_state.chat_initialized = False
    
    # Header
    st.markdown("""
    <div style="margin-top: -3rem; margin-bottom: 0.5rem;">
        <span class="main-header">ADK Agent Manager</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("Interactive interface for testing and visualizing agent execution")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Get available agents
        agents = get_available_agents()
        
        if not agents:
            st.error("No agent configurations found in configs/ directory")
            return
        
        # Agent selector
        agent_names = [agent["name"] for agent in agents]
        selected_agent_name = st.selectbox(
            "Select Agent",
            agent_names,
            help="Choose an agent to test"
        )
        
        # Find selected agent
        selected_agent = next(agent for agent in agents if agent["name"] == selected_agent_name)
        
        st.markdown("---")
        
        # Example queries
        st.markdown("### 💡 Example Queries")
        
        config = selected_agent["config"]
        example_queries = config.get("example_queries", [])
        
        if example_queries:
            # Display example queries from config
            for query in example_queries:
                st.code(query)
        else:
            # Fallback: Generate examples based on tools/agent type if no example_queries defined
            agent_type = config.get("agent_type", "llm")
            
            if "calculator" in config.get("tools", []):
                st.code("What is 15 * 8 + 42?")
            if "WebSearchTool" in config.get("tools", []):
                st.code("Search for Python tutorials")
            if "get_weather" in config.get("tools", []):
                st.code("What's the weather in Tokyo?")
            if agent_type == "sequential":
                st.code("Write a function to sort a list")
            if "greeting" in selected_agent_name.lower():
                st.code("Hello! How are you?")
            if "code_reviewer" in selected_agent_name.lower():
                st.code("Review this code: def add(a,b): return a+b")
            
            # If still no examples shown, show generic ones
            if not config.get("tools", []) and agent_type == "llm" and not any([
                "greeting" in selected_agent_name.lower(),
                "code_reviewer" in selected_agent_name.lower()
            ]):
                st.code("Hello, how can you help me?")
                st.code("Tell me about your capabilities")
        
        st.markdown("---")
        st.markdown("### 📚 About")
        st.markdown("""
        <div style="color: #6b7280; font-size: 0.95rem; line-height: 1.5;">
            This UI allows you to test agents, view their configuration, and see execution results in real-time.
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area - with spacer tabs to push Create Agent to the right
    tab1, tab2, tab3, tab4, spacer1, spacer2, tab5, tab6 = st.tabs([
        "💬 Chat Mode", 
        "🎯 Single Execution", 
        "📊 Agent Details",
        "🔌 MCP Servers",
        "　", 
        "　　", 
        "➕ Create Agent",
        "➕ Create MCP Server"
    ])
    
    # TAB 1: Chat Mode
    with tab1:
        st.markdown("## 💬 Chat with Agent")
        st.markdown("""
        <div style="color: #6b7280; font-size: 0.95rem; margin-bottom: 1.5rem;">
            💡 Chat mode maintains conversation context across multiple messages using a persistent session.
        </div>
        """, unsafe_allow_html=True)
        
        # Check if agent changed - reset if needed
        if st.session_state.chat_agent_name != selected_agent_name:
            if st.session_state.chat_agent_manager:
                asyncio.run(st.session_state.chat_agent_manager.close())
            st.session_state.chat_messages = []
            st.session_state.chat_agent_manager = None
            st.session_state.chat_agent_name = None
            st.session_state.chat_initialized = False
        
        # Chat controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if not st.session_state.chat_initialized:
                if st.button("🚀 Start Chat Session", type="primary", use_container_width=True):
                    with st.spinner("Initializing agent..."):
                        result = asyncio.run(initialize_chat_agent(selected_agent["path"]))
                        
                        if result["success"]:
                            st.session_state.chat_agent_manager = result["agent_manager"]
                            st.session_state.chat_agent_name = selected_agent_name
                            st.session_state.chat_initialized = True
                            st.session_state.chat_messages = []
                            st.success("✅ Chat session started!")
                            st.rerun()
                        else:
                            st.error(f"❌ Failed to initialize agent: {result['error']}")
        
        with col2:
            if st.session_state.chat_initialized:
                if st.button("🔄 New Chat", use_container_width=True):
                    # Cleanup old agent
                    if st.session_state.chat_agent_manager:
                        asyncio.run(st.session_state.chat_agent_manager.close())
                    
                    # Reset session
                    st.session_state.chat_messages = []
                    st.session_state.chat_agent_manager = None
                    st.session_state.chat_agent_name = None
                    st.session_state.chat_initialized = False
                    st.rerun()
        
        with col3:
            if st.session_state.chat_initialized and st.session_state.chat_messages:
                if st.button("🗑️ Clear History", use_container_width=True):
                    st.session_state.chat_messages = []
                    st.rerun()
        
        # Display session info if active
        if st.session_state.chat_initialized and st.session_state.chat_agent_manager:
            with st.expander("🔍 Session Info & Debug"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**User ID:**")
                    st.code(st.session_state.chat_agent_manager.user_id)
                with col2:
                    st.markdown("**Session ID:**")
                    st.code(st.session_state.chat_agent_manager.session_id)
                st.markdown(f"**UI Messages:** {len(st.session_state.chat_messages)}")
                
                # Show ADK session history for debugging
                if st.button("🔍 Check ADK Session History"):
                    with st.spinner("Fetching session history..."):
                        adk_history = asyncio.run(st.session_state.chat_agent_manager.get_conversation_history())
                        st.markdown(f"**ADK Session Turns:** {len(adk_history)}")
                        if adk_history:
                            st.json(adk_history)
                        else:
                            st.warning("No conversation history found in ADK session!")
        
        st.markdown("---")
        
        # Display chat messages
        if st.session_state.chat_initialized:
            # Chat container
            chat_container = st.container()
            
            with chat_container:
                for message in st.session_state.chat_messages:
                    with st.chat_message(message["role"]):
                        # Display JSON or markdown based on message type
                        if message.get("is_json", False):
                            st.json(message["content"])
                        else:
                            st.markdown(message["content"])
                        if "timestamp" in message:
                            st.caption(message["timestamp"])
            
            # Chat input
            if prompt := st.chat_input("Type your message here...", key="chat_input"):
                # Add user message to chat
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                st.session_state.chat_messages.append({
                    "role": "user",
                    "content": prompt,
                    "timestamp": timestamp
                })
                
                # Display user message
                with chat_container:
                    with st.chat_message("user"):
                        st.markdown(prompt)
                        st.caption(timestamp)
                
                # Get agent response
                with st.spinner("🤔 Agent is thinking..."):
                    result = asyncio.run(send_chat_message(st.session_state.chat_agent_manager, prompt))
                    
                    if result["success"]:
                        response_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": result["response"],
                            "is_json": result.get("is_json", False),
                            "timestamp": response_timestamp
                        })
                        
                        # Display assistant message
                        with chat_container:
                            with st.chat_message("assistant"):
                                if result.get("is_json"):
                                    st.json(result["response"])
                                else:
                                    st.markdown(result["response"])
                                st.caption(response_timestamp)
                        
                        st.rerun()
                    else:
                        st.error(f"❌ Error: {result['error']}")
        
        else:
            # Show welcome message
            st.markdown("### 👋 Welcome to Chat Mode")
            st.markdown("""
            Click **Start Chat Session** to begin a conversation with the selected agent.
            
            **Features:**
            - 🔄 Maintains conversation context
            - 💾 Persistent session across multiple messages
            - 🧠 Agent remembers previous interactions
            - 🔧 Access to all agent tools and capabilities
            """)
    
    # TAB 2: Single Execution
    with tab2:
        st.markdown("## 🚀 Single Execution")
        st.markdown("""
        <div style="color: #6b7280; font-size: 0.95rem; margin-bottom: 1rem;">
            💡 Execute a single query without maintaining conversation context.
        </div>
        """, unsafe_allow_html=True)
        
        # Query input
        query = st.text_area(
            "Enter your query:",
            height=100,
            placeholder="Type your message here...",
            help="Enter the message you want to send to the agent"
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            execute_button = st.button("▶️ Execute", type="primary", use_container_width=True)
        
        with col2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_button:
            st.rerun()
        
        # Execution
        if execute_button:
            if not query.strip():
                st.warning("⚠️ Please enter a query")
            else:
                with st.spinner("🔄 Executing agent..."):
                    # Display execution info
                    st.markdown('<div class="execution-box">', unsafe_allow_html=True)
                    st.markdown(f"**Agent:** `{selected_agent_name}`")
                    st.markdown(f"**Time:** `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Execute
                    result = asyncio.run(execute_agent(selected_agent["path"], query))
                    
                    if result["success"]:
                        # Success
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        if result.get("is_json"):
                            st.markdown("### ✅ Execution Successful - Structured JSON Output")
                        else:
                            st.markdown("### ✅ Execution Successful")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Response - Display differently for JSON vs text
                        if result.get("is_json"):
                            st.markdown("### 📊 Structured Response (JSON)")
                            st.json(result["response"])
                        else:
                            st.markdown("### 💬 Agent Response")
                            st.markdown(result["response"])
                        
                        # Execution details
                        with st.expander("🔍 Execution Details"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**User ID:**")
                                st.code(result["user_id"])
                            with col2:
                                st.markdown("**Session ID:**")
                                st.code(result["session_id"])
                            
                            if result.get("has_schema"):
                                st.markdown("""
                                <div style="background-color: #e8e8e8; padding: 0.5rem; border-radius: 0.5rem; color: #262730; font-size: 0.9rem;">
                                    ℹ️ This agent uses structured output (response_schema)
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Download option
                        if result.get("is_json"):
                            download_data = json.dumps(result["response"], indent=2)
                            file_name = f"response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                            mime_type = "application/json"
                        else:
                            download_data = result["response"]
                            file_name = f"response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                            mime_type = "text/plain"
                        
                        st.download_button(
                            label="📥 Download Response",
                            data=download_data,
                            file_name=file_name,
                            mime=mime_type
                        )
                    
                    else:
                        # Error
                        st.error(f"❌ Execution Failed: {result['error']}")
    
    # TAB 3: Agent Details
    with tab3:
        st.markdown("""
        <h2 style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 1.8rem;
                height: 1.8rem;
                border-radius: 50%;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                font-size: 1.2rem;
                font-weight: bold;
                font-style: italic;
            ">i</span>
            <span>Agent Configuration</span>
        </h2>
        """, unsafe_allow_html=True)
        display_agent_metadata(selected_agent)
        
        # Export code section
        st.markdown("---")
        st.markdown("### 🔧 Export as Standalone Code")
        st.markdown("""
        <div style="color: #6b7280; font-size: 0.9rem; margin-bottom: 1rem;">
            Generate standalone Python code for this agent that can be used anywhere without the AgentManager class.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("📤 Generate Code", use_container_width=True):
                st.session_state.show_exported_code = True
                st.session_state.exported_for_agent = selected_agent["name"]
        
        # Reset export view if agent changed
        if st.session_state.get('exported_for_agent') != selected_agent["name"]:
            st.session_state.show_exported_code = False
        
        if st.session_state.get('show_exported_code', False):
            # Generate the code
            exported_code = generate_agent_code(selected_agent["config"])
            
            st.markdown("#### Generated Python Code")
            st.code(exported_code, language="python")
            
            # Action buttons
            col1, col2, col3 = st.columns([1, 1, 3])
            with col1:
                st.download_button(
                    label="💾 Download",
                    data=exported_code,
                    file_name=f"{selected_agent['config'].get('app_name', 'agent')}_standalone.py",
                    mime="text/x-python",
                    use_container_width=True
                )
            with col2:
                if st.button("❌ Hide", use_container_width=True):
                    st.session_state.show_exported_code = False
                    st.rerun()
            
            st.info("💡 This code can be run independently. Make sure to have the tool files available if the agent uses tools.")
        
        # Raw config
        st.markdown("---")
        with st.expander("📄 View Raw Configuration"):
            st.json(selected_agent["config"])
    
    # TAB 4: MCP Servers
    with tab4:
        st.markdown("## 🔌 MCP Server Management")
        st.markdown("""
        <div style="color: #6b7280; font-size: 0.9rem; margin-bottom: 1.5rem;">
            Create and manage MCP (Model Context Protocol) servers with dynamic tool loading.
        </div>
        """, unsafe_allow_html=True)
        
        # Server selector
        available_servers = get_available_mcp_servers()
        
        if not available_servers:
            st.info("📭 No MCP server configurations found. Create one below to get started!")
        else:
            server_names = [s["name"] for s in available_servers]
            selected_server_name = st.selectbox(
                "Select MCP Server",
                options=server_names,
                key="mcp_server_selector"
            )
            
            selected_server = next((s for s in available_servers if s["name"] == selected_server_name), None)
            
            if selected_server:
                server_config = selected_server["config"].get("server", {})
                
                # Display server details
                st.markdown("### 📋 Server Configuration")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**Name:**")
                    st.code(server_config.get("name", "N/A"))
                with col2:
                    st.markdown("**Transport:**")
                    st.code(server_config.get("transport", "stdio"))
                with col3:
                    if server_config.get("transport") in ["http", "sse"]:
                        st.markdown("**Port:**")
                        st.code(server_config.get("port", "N/A"))
                    else:
                        st.markdown("**Mode:**")
                        st.code("Local (stdio)")
                
                st.markdown("**Description:**")
                st.markdown(f"""
                <div style="background-color: #e8e8e8; padding: 1rem; border-radius: 0.5rem; color: #262730; margin-bottom: 1rem;">
                    {server_config.get("description", "No description provided")}
                </div>
                """, unsafe_allow_html=True)
                
                # Display tools
                st.markdown("### 🛠️ Configured Tools")
                tools = selected_server["config"].get("tools", [])
                if tools:
                    for tool in tools:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**{tool.get('file')}**")
                            st.markdown(f"<div style='color: #6b7280; font-size: 0.85rem;'>{tool.get('type')} - {tool.get('function_name', tool.get('class_name', 'N/A'))}</div>", unsafe_allow_html=True)
                        with col2:
                            st.markdown(f"`{tool.get('type')}`")
                else:
                    st.info("No tools configured")
                
                st.markdown("---")
                
                # Server controls
                st.markdown("### 🎮 Server Controls")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("🚀 Initialize Server", use_container_width=True):
                        try:
                            with st.spinner("Initializing MCP server..."):
                                server_manager = MCPServerManager(selected_server["path"])
                                server_manager.initialize()
                            st.success("✅ Server initialized successfully!")
                            st.info("💡 To start the server, run it from the command line or use the generated code below.")
                        except Exception as e:
                            st.error(f"❌ Initialization failed: {e}")
                
                with col2:
                    if st.button("📋 View Config JSON", use_container_width=True):
                        st.json(selected_server["config"])
                
                with col3:
                    if st.button("📥 Export for Claude", use_container_width=True):
                        try:
                            server_manager = MCPServerManager(selected_server["path"])
                            claude_config = server_manager.export_config_for_claude()
                            st.json(claude_config)
                            st.markdown("""
                            <div style="background-color: #e8e8e8; padding: 1rem; border-radius: 0.5rem; color: #262730; font-size: 0.85rem; margin-top: 1rem;">
                                💡 Add this configuration to your Claude Desktop config file:<br>
                                <code>~/Library/Application Support/Claude/claude_desktop_config.json</code> (Mac)<br>
                                <code>%APPDATA%\\Claude\\claude_desktop_config.json</code> (Windows)
                            </div>
                            """, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"❌ Export failed: {e}")
                
                st.markdown("---")
                
                # Usage instructions
                st.markdown("### 📖 Usage Instructions")
                st.markdown("""
                <div style="background-color: #e8e8e8; padding: 1rem; border-radius: 0.5rem; color: #262730;">
                    <strong>Run this server:</strong><br><br>
                    <code>python -m mcp_def.mcp_manager configs_mcp/{filename}</code>
                </div>
                """.format(filename=Path(selected_server["path"]).name), unsafe_allow_html=True)
    
    # Spacer tabs (empty)
    with spacer1:
        st.empty()
    with spacer2:
        st.empty()
    
    # TAB 5: Create Agent
    with tab5:
        st.markdown("## ➕ Create New Agent")
        st.markdown("""
        <div style="color: #6b7280; font-size: 0.95rem; margin-bottom: 1.5rem;">
            Fill in the form below to create a new agent configuration. The agent will be saved as a JSON file in the configs/ directory.
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize session state for generated instruction
        if 'generated_instruction' not in st.session_state:
            st.session_state.generated_instruction = ""
        if 'form_agent_name' not in st.session_state:
            st.session_state.form_agent_name = ""
        if 'form_description' not in st.session_state:
            st.session_state.form_description = ""
        if 'use_structured_output' not in st.session_state:
            st.session_state.use_structured_output = False
        
        # Structured output toggle (outside form for immediate feedback)
        st.markdown("#### Response Schema (Optional)")
        use_schema_toggle = st.checkbox(
            "Enable Structured Output (JSON)",
            value=st.session_state.use_structured_output,
            help="Define a JSON schema for consistent, parseable responses",
            key="schema_toggle"
        )
        st.session_state.use_structured_output = use_schema_toggle
        
        st.markdown("")  # Add spacing
        
        with st.form("create_agent_form"):
            st.markdown("### Basic Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                app_name = st.text_input(
                    "App Name *",
                    help="Unique identifier for the agent. Spaces will be replaced with underscores (e.g., 'customer_support')",
                    placeholder="my_agent"
                )
                
                # Show the filename that will be created
                if app_name:
                    safe_name = app_name.replace(" ", "_")
                    st.caption(f"📄 Will be saved as: `{safe_name}.json`")
                
                agent_name = st.text_input(
                    "Agent Name *",
                    value=st.session_state.form_agent_name,
                    help="Name for the agent. Spaces will be replaced with underscores for the identifier",
                    placeholder="My Custom Agent"
                )
                
                # Show the identifier that will be used
                if agent_name:
                    safe_name = agent_name.replace(" ", "_")
                    st.caption(f"🔖 Agent identifier: `{safe_name}`")
                
                agent_type = st.selectbox(
                    "Agent Type *",
                    options=["llm", "sequential"],
                    help="LLM for intelligent agents, Sequential for workflows"
                )
            
            with col2:
                model = st.selectbox(
                    "Model *",
                    options=[
                        "gemini-2.5-flash",
                        "gemini-2.5-pro"
                    ],
                    format_func=lambda x: {
                        "gemini-2.5-flash": "Gemini 2.5 Flash",
                        "gemini-2.5-pro": "Gemini 2.5 Pro"
                    }.get(x, x),
                    help="AI model to use"
                )
                
                output_key = st.text_input(
                    "Output Key (Optional)",
                    help="For sequential workflows - key to store output in state",
                    placeholder="agent_output"
                )
            
            st.markdown("### Agent Behavior")
            
            description = st.text_area(
                "Description *",
                value=st.session_state.form_description,
                help="Brief description of what the agent does",
                placeholder="This agent helps with...",
                height=80
            )
            
            # Instructions section with AI generation helper
            instruction = st.text_area(
                "Instructions *",
                value=st.session_state.generated_instruction,
                help="System instructions that define the agent's behavior",
                placeholder="You are an expert assistant that...",
                height=150
            )
            
            generate_btn = st.form_submit_button("✨ Generate with AI", use_container_width=False)
            
            # Status placeholder that appears right below the button
            status_placeholder = st.empty()
            
            st.markdown("### Tools & Sub-Agents (Optional)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Get available tools
                tools_dir = Path("tools")
                available_tools = []
                if tools_dir.exists():
                    for tool_file in tools_dir.glob("*.py"):
                        if tool_file.stem != "__init__":
                            available_tools.append(tool_file.stem)
                
                selected_tools = st.multiselect(
                    "Tools",
                    options=available_tools,
                    help="Select tools this agent can use"
                )
            
            with col2:
                # Get available sub-agents
                configs_dir = Path("configs")
                available_agents = []
                if configs_dir.exists():
                    for config_file in configs_dir.glob("*.json"):
                        available_agents.append(config_file.stem)
                
                selected_sub_agents = st.multiselect(
                    "Sub-Agents",
                    options=available_agents,
                    help="Select sub-agents (for LLM type) or workflow steps (for Sequential type)"
                )
            
            st.markdown("---")
            st.markdown("### 🔌 MCP Servers (Optional)")
            st.markdown("""
            <div style="color: #6b7280; font-size: 0.9rem; margin-bottom: 1rem;">
                Connect this agent to MCP servers to use their tools. Select servers from your configs_mcp/ folder.
            </div>
            """, unsafe_allow_html=True)
            
            # Get available MCP servers
            configs_mcp_dir = Path("configs_mcp")
            available_mcp_servers = []
            if configs_mcp_dir.exists():
                for config_file in configs_mcp_dir.glob("*.json"):
                    available_mcp_servers.append(config_file.stem)
            
            if available_mcp_servers:
                selected_mcp_servers = st.multiselect(
                    "Select MCP Servers",
                    options=available_mcp_servers,
                    help="Choose which MCP servers this agent should connect to"
                )
                
                # Configure each selected MCP server
                mcp_server_configs = []
                if selected_mcp_servers:
                    st.markdown("**Configure Selected MCP Servers:**")
                    for server_name in selected_mcp_servers:
                        with st.expander(f"⚙️ {server_name}"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                server_url = st.text_input(
                                    "Server URL (Optional)",
                                    key=f"mcp_url_{server_name}",
                                    placeholder="e.g., http://localhost:8000",
                                    help="Leave empty to auto-start the server"
                                )
                            
                            with col2:
                                auto_start = st.checkbox(
                                    "Auto-start server",
                                    key=f"mcp_auto_start_{server_name}",
                                    value=True if not server_url else False,
                                    help="Automatically start this MCP server when the agent initializes"
                                )
                            
                            # Build MCP server config entry
                            server_config = {
                                "config": server_name
                            }
                            if server_url:
                                server_config["url"] = server_url
                            if auto_start:
                                server_config["auto_start"] = True
                            
                            mcp_server_configs.append(server_config)
            else:
                st.info("💡 No MCP servers found in configs_mcp/ folder. Create one in the 'Create MCP Server' tab first.")
                selected_mcp_servers = []
                mcp_server_configs = []
            
            st.markdown("---")
            st.markdown("### 💡 Example Queries (Optional)")
            st.markdown("""
            <div style="color: #6b7280; font-size: 0.9rem; margin-bottom: 1rem;">
                Add example queries that users can try with this agent. These will be displayed in the UI.
            </div>
            """, unsafe_allow_html=True)
            
            # Number of example queries
            num_examples = st.number_input(
                "Number of example queries",
                min_value=0,
                max_value=10,
                value=3,
                help="How many example queries to add?"
            )
            
            example_queries = []
            for i in range(int(num_examples)):
                query = st.text_input(
                    f"Example Query {i+1}",
                    key=f"example_query_{i}",
                    placeholder="e.g., What's the weather in Tokyo?",
                    help="An example query that demonstrates this agent's capabilities"
                )
                if query:
                    example_queries.append(query)
            
            # Schema builder section (appears if enabled via toggle outside form)
            schema_fields = []
            schema_json = ""
            
            if st.session_state.use_structured_output:
                st.markdown("---")
                st.markdown("### 📋 Define Output Fields")
                st.markdown("""
                <div style="color: #6b7280; font-size: 0.9rem; margin-bottom: 1rem;">
                    Add fields that your agent will return in JSON format. Each field will have a name, type, and description.
                </div>
                """, unsafe_allow_html=True)
                
                # Number of fields to add
                num_fields = st.number_input(
                    "Number of fields",
                    min_value=1,
                    max_value=20,
                    value=3,
                    help="How many fields should the output have?"
                )
                
                # Collect field definitions
                for i in range(int(num_fields)):
                    st.markdown(f"**Field {i+1}**")
                    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                    
                    with col1:
                        field_name = st.text_input(
                            f"Field Name",
                            key=f"field_name_{i}",
                            placeholder="e.g., product_name",
                            help="Name of the field (use snake_case)"
                        )
                    
                    with col2:
                        field_type = st.selectbox(
                            f"Type",
                            options=["string", "integer", "number", "boolean", "array", "object"],
                            key=f"field_type_{i}",
                            help="Data type for this field"
                        )
                    
                    with col3:
                        field_desc = st.text_input(
                            f"Description",
                            key=f"field_desc_{i}",
                            placeholder="What this field contains",
                            help="Help the AI understand what to put here"
                        )
                    
                    with col4:
                        field_required = st.checkbox(
                            "Required",
                            key=f"field_req_{i}",
                            value=True,
                            help="Is this field mandatory?"
                        )
                    
                    # Handle array items
                    array_item_type = "string"
                    if field_type == "array":
                        array_item_type = st.selectbox(
                            f"Array items are",
                            options=["string", "integer", "number", "boolean", "object"],
                            key=f"array_item_type_{i}",
                            help="What type of items does this array contain?"
                        )
                    
                    # Handle enum for strings
                    enum_values = []
                    if field_type == "string":
                        use_enum = st.checkbox(
                            f"Limit to specific values (enum)?",
                            key=f"use_enum_{i}",
                            help="Restrict to a predefined list of values"
                        )
                        if use_enum:
                            enum_input = st.text_input(
                                f"Allowed values (comma-separated)",
                                key=f"enum_values_{i}",
                                placeholder="e.g., positive, neutral, negative",
                                help="Enter allowed values separated by commas"
                            )
                            if enum_input:
                                enum_values = [v.strip() for v in enum_input.split(",")]
                    
                    if field_name:
                        schema_fields.append({
                            "name": field_name,
                            "type": field_type,
                            "description": field_desc,
                            "required": field_required,
                            "array_item_type": array_item_type if field_type == "array" else None,
                            "enum": enum_values if enum_values else None
                        })
                    
                    st.markdown("---")
            
            # Submit button
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                submitted = st.form_submit_button("💾 Create Agent", type="primary", use_container_width=True)
            
            with col2:
                preview = st.form_submit_button("👁️ Preview", use_container_width=True)
        
        # Handle AI generation button
        if generate_btn:
            if not agent_name or not description:
                status_placeholder.error("❌ Please fill in Agent Name and Description first before generating instructions.")
            else:
                # Show loading in the status placeholder
                with status_placeholder:
                    with st.spinner("🤖 Generating instructions with AI..."):
                        try:
                            # Define async function for prompt generation
                            async def generate_prompt():
                                # Initialize the prompt generator agent
                                prompt_gen_manager = AgentManager("configs/helper_prompt_generator_agent.json")
                                await prompt_gen_manager.initialize()
                                
                                # Create query for the generator
                                query = f"Agent Name: {agent_name}\n\nDescription: {description}\n\nGenerate comprehensive system instructions for this agent."
                                
                                # Get generated instructions
                                generated_instruction = await prompt_gen_manager.send_message(query)
                                
                                # Clean up
                                await prompt_gen_manager.close()
                                
                                return generated_instruction
                            
                            # Run the async function
                            generated_instruction = asyncio.run(generate_prompt())
                            
                            # Store in session state and rerun to populate the form
                            st.session_state.generated_instruction = generated_instruction
                            st.session_state.form_agent_name = agent_name
                            st.session_state.form_description = description
                            
                            st.rerun()
                            
                        except Exception as e:
                            status_placeholder.error(f"❌ Failed to generate instructions: {str(e)}")
        
        # Handle form submission
        if submitted or preview:
            # Validate required fields
            errors = []
            if not app_name:
                errors.append("App Name is required")
            if not agent_name:
                errors.append("Agent Name is required")
            if not description:
                errors.append("Description is required")
            if not instruction:
                errors.append("Instructions are required")
            if agent_type == "sequential" and not selected_sub_agents:
                errors.append("Sequential agents require at least one sub-agent")
            
            if errors:
                for error in errors:
                    st.error(f"❌ {error}")
            else:
                # Build the configuration (replace spaces for valid identifiers)
                safe_app_name = app_name.replace(" ", "_")
                safe_agent_name = agent_name.replace(" ", "_")
                
                config = {
                    "app_name": safe_app_name,
                    "agent_type": agent_type,
                    "agent": {
                        "name": safe_agent_name,
                        "description": description,
                        "instruction": instruction
                    },
                    "tools": selected_tools if selected_tools else [],
                    "sub_agents": selected_sub_agents if selected_sub_agents else []
                }
                
                # Add model for LLM agents
                if agent_type == "llm":
                    config["agent"]["model"] = model
                
                # Add output_key if provided
                if output_key:
                    config["agent"]["output_key"] = output_key
                
                # Add response schema if provided
                if st.session_state.use_structured_output and schema_fields:
                    # Build the JSON schema from fields
                    properties = {}
                    required_fields = []
                    
                    for field in schema_fields:
                        field_schema = {
                            "type": field["type"],
                            "description": field["description"]
                        }
                        
                        # Handle array type
                        if field["type"] == "array" and field["array_item_type"]:
                            field_schema["items"] = {"type": field["array_item_type"]}
                        
                        # Handle enum
                        if field["enum"]:
                            field_schema["enum"] = field["enum"]
                        
                        properties[field["name"]] = field_schema
                        
                        if field["required"]:
                            required_fields.append(field["name"])
                    
                    schema_dict = {
                        "type": "object",
                        "properties": properties,
                        "required": required_fields
                    }
                    
                    config["agent"]["response_schema"] = schema_dict
                
                # Add example queries if provided
                if example_queries:
                    config["example_queries"] = example_queries
                
                # Add MCP servers if provided
                if mcp_server_configs:
                    config["mcp_servers"] = mcp_server_configs
                
                # Preview mode
                if preview:
                    st.markdown("### 👁️ Configuration Preview")
                    st.json(config)
                
                # Save mode
                if submitted:
                    try:
                        # Ensure configs directory exists
                        configs_dir = Path("configs")
                        configs_dir.mkdir(exist_ok=True)
                        
                        # Create filename using the safe app name from config
                        filename = f"{config['app_name']}.json"
                        filepath = configs_dir / filename
                        
                        # Check if file exists
                        if filepath.exists():
                            st.error(f"❌ Agent configuration '{filename}' already exists. Choose a different app name.")
                        else:
                            # Save the file
                            with open(filepath, 'w') as f:
                                json.dump(config, f, indent=2)
                            
                            # Clear form session state
                            st.session_state.generated_instruction = ""
                            st.session_state.form_agent_name = ""
                            st.session_state.form_description = ""
                            
                            st.success(f"✅ Agent configuration saved as '{filename}'")
                            st.info("🔄 Refresh the page to see your new agent in the agent selector!")
                            
                            # Show the config
                            st.markdown("### 📄 Created Configuration")
                            st.json(config)
                            
                            # Add a rerun button
                            if st.button("🔄 Reload UI"):
                                st.rerun()
                    
                    except Exception as e:
                        st.error(f"❌ Error saving configuration: {e}")
    
    # TAB 6: Create MCP Server
    with tab6:
        st.markdown("## ➕ Create New MCP Server")
        st.markdown("""
        <div style="color: #6b7280; font-size: 0.95rem; margin-bottom: 1.5rem;">
            Configure a new MCP server with custom tools.
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize session state for function code generation
        if 'generated_function_code' not in st.session_state:
            st.session_state.generated_function_code = ""
        if 'function_code_key' not in st.session_state:
            st.session_state.function_code_key = 0
        
        # Write Function Section (outside form for immediate interaction)
        st.markdown("### ✍️ Write Function (Optional)")
        st.markdown("""
        <div style="color: #6b7280; font-size: 0.9rem; margin-bottom: 1rem;">
            Create a custom function that will be saved to the tools/ folder and made available for selection below.
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("➕ Create New Function", expanded=False):
            func_col1, func_col2 = st.columns(2)
            
            with func_col1:
                function_name = st.text_input(
                    "Function Name",
                    placeholder="calculate_distance",
                    help="Python function name (snake_case, no spaces)",
                    key="mcp_function_name"
                )
            
            with func_col2:
                function_file_name = st.text_input(
                    "File Name",
                    value=function_name if function_name else "",
                    placeholder="calculate_distance",
                    help="Filename for the tool (without .py extension)",
                    key="mcp_function_file_name"
                )
            
            function_description = st.text_area(
                "Function Description",
                placeholder="Calculate the distance between two points using the Euclidean formula",
                help="Describe what this function should do. Used for AI generation and docstring.",
                height=100,
                key="mcp_function_description"
            )
            
            # AI Generation Button
            ai_gen_col1, ai_gen_col2 = st.columns([1, 4])
            with ai_gen_col1:
                generate_function_btn = st.button("✨ Generate with AI", key="generate_function_ai", use_container_width=True)
            
            # Status placeholder for AI generation
            function_status_placeholder = st.empty()
            
            # Handle AI generation
            if generate_function_btn:
                if not function_name or not function_description:
                    function_status_placeholder.error("❌ Please provide both function name and description")
                else:
                    with function_status_placeholder:
                        with st.spinner("🤖 Generating function code..."):
                            try:
                                # Call helper agent to generate function code
                                helper_manager = AgentManager("configs/helper_prompt_generator_agent.json")
                                asyncio.run(helper_manager.initialize())
                                
                                prompt = f"""Generate a Python function with the following specifications:

Function Name: {function_name}
Description: {function_description}

Requirements:
1. Write ONLY the Python function code (no explanations, no markdown)
2. Include proper type hints for parameters and return type
3. Include a detailed docstring in Google style
4. Add input validation where appropriate
5. Include 1-2 example uses in the docstring
6. Make the function ready to be imported and used as an MCP tool
7. If the function needs external libraries, add a comment at the top listing them

Format the output as clean, executable Python code."""

                                response = asyncio.run(helper_manager.send_message(prompt, return_json=False))
                                
                                # Clean up the response (remove markdown code blocks if present)
                                generated_code = response.strip()
                                if generated_code.startswith("```python"):
                                    generated_code = generated_code.split("```python")[1]
                                if generated_code.startswith("```"):
                                    generated_code = generated_code.split("```")[1]
                                if "```" in generated_code:
                                    generated_code = generated_code.split("```")[0]
                                
                                st.session_state.generated_function_code = generated_code.strip()
                                st.session_state.function_code_key += 1
                                
                                helper_manager.close()
                                function_status_placeholder.success("✅ Function code generated!")
                                
                            except Exception as e:
                                function_status_placeholder.error(f"❌ Error generating function: {e}")
            
            # Code Editor
            st.markdown("**Function Code:**")
            function_code = st.text_area(
                "Edit your function code",
                value=st.session_state.generated_function_code,
                height=300,
                help="Write or edit your Python function code here",
                key=f"function_code_editor_{st.session_state.function_code_key}",
                label_visibility="collapsed"
            )
            
            # Save Function Button
            save_func_col1, save_func_col2 = st.columns([1, 4])
            with save_func_col1:
                save_function_btn = st.button("💾 Save Function", key="save_function_btn", use_container_width=True)
            
            if save_function_btn:
                if not function_file_name or not function_code:
                    st.error("❌ Please provide both filename and function code")
                else:
                    try:
                        # Ensure tools directory exists
                        tools_dir = Path("tools")
                        tools_dir.mkdir(exist_ok=True)
                        
                        # Clean filename
                        clean_func_filename = function_file_name.replace(" ", "_").lower()
                        if not clean_func_filename.endswith(".py"):
                            clean_func_filename += ".py"
                        
                        func_filepath = tools_dir / clean_func_filename
                        
                        if func_filepath.exists():
                            st.error(f"❌ Tool file '{clean_func_filename}' already exists")
                        else:
                            # Write the function code to file
                            with open(func_filepath, 'w', encoding='utf-8') as f:
                                f.write(function_code)
                            
                            st.success(f"✅ Function saved as 'tools/{clean_func_filename}'")
                            st.info("💡 Click the '🔄 Refresh Tools List' button below to see your new function in the tool selection.")
                            
                            # Clear the generated code
                            st.session_state.generated_function_code = ""
                            
                    except Exception as e:
                        st.error(f"❌ Error saving function: {e}")
        
        st.markdown("---")
        
        # Refresh tools button
        if st.button("🔄 Refresh Tools List", key="refresh_tools_btn"):
            st.rerun()
        
        with st.form("create_mcp_server_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                server_name = st.text_input(
                    "Server Name",
                    placeholder="My Calculator Server",
                    help="Human-readable name for your MCP server"
                )
            
            with col2:
                file_name = st.text_input(
                    "Config File Name",
                    placeholder="my_calculator_server",
                    help="Filename for the JSON config (without .json extension)"
                )
            
            server_description = st.text_area(
                "Description",
                placeholder="MCP server for calculator operations",
                help="Describe what this server does"
            )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                transport = st.selectbox(
                    "Transport Protocol",
                    options=["stdio", "http", "sse"],
                    help="stdio for local/IDE integration, http/sse for network access"
                )
            
            with col2:
                if transport in ["http", "sse"]:
                    host = st.text_input("Host", value="0.0.0.0")
                else:
                    host = "0.0.0.0"
                    st.text_input("Host", value="0.0.0.0", disabled=True, help="Not used for stdio transport")
            
            with col3:
                if transport in ["http", "sse"]:
                    port = st.number_input("Port", min_value=1024, max_value=65535, value=8000)
                else:
                    port = 8000
                    st.number_input("Port", value=8000, disabled=True, help="Not used for stdio transport")
            
            # Tool selection
            st.markdown("### 🛠️ Select Tools")
            available_tools = get_available_tools()
            
            if available_tools:
                selected_tool_names = st.multiselect(
                    "Tools to include",
                    options=[t["name"] for t in available_tools],
                    help="Select tools to include in this MCP server"
                )
                
                # Show selected tools with their types
                if selected_tool_names:
                    st.markdown("**Selected Tools:**")
                    for tool_name in selected_tool_names:
                        tool = next((t for t in available_tools if t["name"] == tool_name), None)
                        if tool:
                            st.markdown(f"- `{tool['file']}` ({tool['type']}: {tool.get('function_name', tool.get('class_name'))})")
            else:
                st.warning("No tools found in the tools/ directory")
                selected_tool_names = []
            
            col1, col2 = st.columns([1, 4])
            
            with col1:
                submitted = st.form_submit_button("💾 Create Server", use_container_width=True)
            with col2:
                preview = st.form_submit_button("👁️ Preview Config", use_container_width=True)
            
            if preview or submitted:
                # Build config
                config = {
                    "server": {
                        "name": server_name,
                        "description": server_description,
                        "port": port,
                        "host": host,
                        "transport": transport
                    },
                    "tools": []
                }
                
                # Add selected tools
                for tool_name in selected_tool_names:
                    tool = next((t for t in available_tools if t["name"] == tool_name), None)
                    if tool:
                        tool_config = {
                            "file": tool["file"],
                            "type": tool["type"]
                        }
                        if tool["type"] == "function":
                            tool_config["function_name"] = tool["function_name"]
                        else:
                            tool_config["class_name"] = tool["class_name"]
                        config["tools"].append(tool_config)
                
                if preview:
                    st.markdown("### 👁️ Configuration Preview")
                    st.json(config)
                
                if submitted:
                    if not server_name or not file_name:
                        st.error("❌ Server name and filename are required")
                    elif not selected_tool_names:
                        st.error("❌ Please select at least one tool")
                    else:
                        try:
                            # Ensure configs_mcp directory exists
                            configs_dir = Path("configs_mcp")
                            configs_dir.mkdir(exist_ok=True)
                            
                            # Clean filename
                            clean_filename = file_name.replace(" ", "_").lower()
                            filepath = configs_dir / f"{clean_filename}.json"
                            
                            if filepath.exists():
                                st.error(f"❌ Server configuration '{clean_filename}.json' already exists")
                            else:
                                with open(filepath, 'w') as f:
                                    json.dump(config, f, indent=2)
                                
                                st.success(f"✅ MCP server configuration saved as '{clean_filename}.json'")
                                st.info("💡 Reload this page to see the new server in the list above")
                                
                                if st.button("🔄 Reload UI"):
                                    st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error saving configuration: {e}")


if __name__ == "__main__":
    main()

