import uuid
import asyncio
import json
import os
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Optional
from io import StringIO
from dotenv import load_dotenv

from google.adk.agents import Agent, SequentialAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset

from agent_def.tool_loader import ToolLoader
from mcp_def.mcp_manager import MCPServerManager

# Load environment variables from .env file
load_dotenv()

# Suppress deprecation warnings from Google ADK
warnings.filterwarnings('ignore', category=DeprecationWarning, module='google.adk')


@contextmanager
def suppress_adk_warnings():
    """
    Context manager to suppress benign print/warning messages from Google ADK.
    Specifically filters out the "App name mismatch detected" message.
    """
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_capture = StringIO()
    stderr_capture = StringIO()
    sys.stdout = stdout_capture
    sys.stderr = stderr_capture
    
    try:
        yield
    finally:
        captured_out = stdout_capture.getvalue()
        captured_err = stderr_capture.getvalue()
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
        # Print everything except the app name mismatch line
        for line in captured_out.splitlines():
            if "App name mismatch detected" not in line:
                print(line)
        
        for line in captured_err.splitlines():
            if "App name mismatch detected" not in line:
                print(line, file=sys.stderr)


class AgentManager:
    """
    Manages the full lifecycle of an agent including configuration loading,
    session management, and conversation handling.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the AgentManager with a configuration file.
        
        Args:
            config_path: Path to the JSON or YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        # Auto-generate unique IDs for this instance
        self.user_id = str(uuid.uuid4())
        self.session_id = str(uuid.uuid4())
        self.app_name = self.config.get("app_name", "default_agent")
        
        # Initialize session service
        self.session_service = InMemorySessionService()
        
        # Initialize tool loader
        self.tool_loader = ToolLoader()
        
        # These will be set during initialize()
        self.agent: Optional[Agent] = None
        self.runner: Optional[Runner] = None
        self.session = None
        self.sub_agent_instances = []  # Store sub-agent Agent objects for cleanup
        
        # MCP server integration
        self.mcp_servers = []  # List of MCPServerManager instances for auto-started servers
        self.mcp_toolsets = []  # List of MCPToolset instances
        
        print(f"AgentManager created: App='{self.app_name}', User='{self.user_id}', Session='{self.session_id}'")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from JSON or YAML file."""
        path = Path(config_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(path, 'r') as f:
            if path.suffix == '.json':
                config = json.load(f)
            elif path.suffix in ['.yaml', '.yml']:
                try:
                    import yaml
                    config = yaml.safe_load(f)
                except ImportError:
                    raise ImportError("PyYAML is required for YAML config files. Install with: pip install pyyaml")
            else:
                raise ValueError(f"Unsupported config file format: {path.suffix}. Use .json or .yaml")
        
        print(f"Configuration loaded from: {config_path}")
        return config
    
    def _create_pydantic_model_from_schema(self, schema: dict, model_name: str = "DynamicModel"):
        """
        Convert a JSON schema to a Pydantic model dynamically.
        
        Args:
            schema: JSON schema dictionary
            model_name: Name for the generated Pydantic model
            
        Returns:
            Pydantic model class
        """
        from pydantic import create_model, Field
        from typing import List, Optional, Any
        
        if schema.get("type") != "object":
            raise ValueError("Root schema must be of type 'object'")
        
        properties = schema.get("properties", {})
        required_fields = schema.get("required", [])
        
        # Build field definitions for create_model
        field_definitions = {}
        
        for field_name, field_schema in properties.items():
            field_type = self._json_type_to_python_type(field_schema)
            field_description = field_schema.get("description", "")
            
            # Determine if field is required
            is_required = field_name in required_fields
            
            if is_required:
                field_definitions[field_name] = (
                    field_type,
                    Field(description=field_description)
                )
            else:
                field_definitions[field_name] = (
                    Optional[field_type],
                    Field(default=None, description=field_description)
                )
        
        # Create the Pydantic model dynamically
        return create_model(model_name, **field_definitions)
    
    def _json_type_to_python_type(self, field_schema: dict):
        """Convert JSON schema type to Python type annotation."""
        from typing import List, Optional, Any
        
        json_type = field_schema.get("type")
        
        if json_type == "string":
            # Check for enum
            if "enum" in field_schema:
                from typing import Literal
                # Literal needs unpacked values, not a tuple
                enum_values = field_schema["enum"]
                if len(enum_values) == 1:
                    return Literal[enum_values[0]]
                # For multiple values, just use str since Literal syntax is complex dynamically
                return str
            return str
        elif json_type == "integer":
            return int
        elif json_type == "number":
            return float
        elif json_type == "boolean":
            return bool
        elif json_type == "array":
            items_schema = field_schema.get("items", {})
            item_type = self._json_type_to_python_type(items_schema)
            return List[item_type]
        elif json_type == "object":
            # Nested objects become dict for simplicity
            # Could recursively create models here for more complex cases
            return dict
        else:
            return Any
    
    async def _load_sub_agents(self, sub_agent_names: list) -> list:
        """
        Load and initialize sub-agents from config names.
        
        Args:
            sub_agent_names: List of sub-agent config names (e.g., ["greeting_agent", "farewell_agent"])
            
        Returns:
            List of initialized Agent objects
        """
        sub_agents = []
        
        for sub_agent_name in sub_agent_names:
            try:
                # Build config path (assumes configs/ directory)
                config_path = f"configs/{sub_agent_name}.json"
                
                # Load sub-agent config
                sub_agent_config = self._load_config(config_path)
                
                # Load tools for sub-agent if specified
                sub_tool_names = sub_agent_config.get("tools", [])
                sub_loaded_tools = []
                if sub_tool_names:
                    sub_tool_loader = ToolLoader()
                    sub_loaded_tools = sub_tool_loader.load_tools(sub_tool_names)
                    print(f"  Sub-agent '{sub_agent_name}': Loaded {len(sub_loaded_tools)} tool(s)")
                
                # Recursively load sub-agents of this sub-agent (if any)
                nested_sub_agent_names = sub_agent_config.get("sub_agents", [])
                nested_sub_agents = []
                if nested_sub_agent_names:
                    print(f"  Loading nested sub-agents for '{sub_agent_name}'...")
                    nested_sub_agents = await self._load_sub_agents(nested_sub_agent_names)
                
                # Create sub-agent
                agent_config = sub_agent_config.get("agent", {})
                
                # Build Agent kwargs
                agent_kwargs = {
                    "name": agent_config.get("name", sub_agent_name),
                    "model": agent_config.get("model", "gemini-2.5-flash"),
                    "description": agent_config.get("description", ""),
                    "instruction": agent_config.get("instruction", ""),
                    "tools": sub_loaded_tools,
                }
                
                # Add output_key if specified (for sequential workflows)
                if "output_key" in agent_config:
                    agent_kwargs["output_key"] = agent_config["output_key"]
                
                # Only add sub_agents if there are any
                if nested_sub_agents:
                    agent_kwargs["sub_agents"] = nested_sub_agents
                
                sub_agent = Agent(**agent_kwargs)
                
                sub_agents.append(sub_agent)
                self.sub_agent_instances.append(sub_agent)
                print(f"  Sub-agent loaded: {sub_agent.name}")
                
            except Exception as e:
                print(f"  Warning: Could not load sub-agent '{sub_agent_name}': {e}")
        
        return sub_agents
    
    async def _load_mcp_servers(self):
        """
        Load and connect to MCP servers from config.
        Supports both config file references and direct URLs.
        Can auto-start servers if needed.
        
        Returns:
            List of MCPToolset instances
        """
        mcp_config = self.config.get("mcp_servers", [])
        if not mcp_config:
            return []
        
        print(f"Loading {len(mcp_config)} MCP server(s)...")
        mcp_toolsets = []
        
        for idx, server_entry in enumerate(mcp_config):
            try:
                config_ref = server_entry.get("config")
                url = server_entry.get("url")
                auto_start = server_entry.get("auto_start", False)
                
                # Validation: at least one of config or url must be provided
                if not config_ref and not url:
                    print(f"  Warning: MCP server entry {idx} missing both 'config' and 'url', skipping")
                    continue
                
                server_url = None
                server_manager = None
                
                # Handle auto-start scenario
                if auto_start and config_ref:
                    print(f"  Auto-starting MCP server from config: {config_ref}")
                    try:
                        config_path = f"configs_mcp/{config_ref}.json"
                        server_manager = MCPServerManager(config_path)
                        
                        # Initialize the server
                        server_manager.initialize()
                        
                        # Check if we need to ensure HTTP transport for auto-start
                        if server_manager.transport == "stdio":
                            print(f"    Warning: Cannot auto-start stdio server, use http/sse transport")
                            print(f"    Skipping auto-start for {config_ref}")
                            continue
                        
                        # Start the server in background
                        server_manager.start_server()
                        
                        # Store the server manager so we can stop it later
                        self.mcp_servers.append(server_manager)
                        
                        # Give server a moment to start
                        time.sleep(0.5)
                        
                        # Determine URL from config
                        server_url = url or f"http://{server_manager.host}:{server_manager.port}"
                        print(f"    ✓ Server started at: {server_url}")
                        
                    except Exception as e:
                        print(f"    ✗ Failed to auto-start server '{config_ref}': {e}")
                        continue
                
                # If we didn't auto-start, determine URL
                if not server_url:
                    if url:
                        # Direct URL provided
                        server_url = url
                    elif config_ref:
                        # Load config to get URL
                        config_path = f"configs_mcp/{config_ref}.json"
                        if Path(config_path).exists():
                            with open(config_path, 'r') as f:
                                mcp_server_config = json.load(f)
                                server_config = mcp_server_config.get("server", {})
                                host = server_config.get("host", "localhost")
                                port = server_config.get("port", 8000)
                                server_url = f"http://{host}:{port}"
                        else:
                            print(f"  Warning: Config file not found: {config_path}")
                            continue
                
                # Create MCPToolset
                print(f"  Connecting to MCP server at: {server_url}")
                mcp_toolset = MCPToolset(url=server_url)
                mcp_toolsets.append(mcp_toolset)
                self.mcp_toolsets.append(mcp_toolset)
                print(f"  ✓ Connected to MCP server: {server_url}")
                
            except Exception as e:
                print(f"  Warning: Failed to load MCP server entry {idx}: {e}")
                continue
        
        print(f"Loaded {len(mcp_toolsets)} MCP toolset(s)")
        return mcp_toolsets
    
    async def initialize(self):
        """
        Initialize the agent, session, and runner.
        This must be called before sending messages.
        """
        # Load tools if specified in config
        tool_names = self.config.get("tools", [])
        loaded_tools = []
        if tool_names:
            try:
                loaded_tools = self.tool_loader.load_tools(tool_names)
                print(f"Loaded {len(loaded_tools)} tool(s)")
            except Exception as e:
                print(f"Warning: Error loading tools: {e}")
        
        # Load sub-agents if specified in config
        sub_agent_names = self.config.get("sub_agents", [])
        loaded_sub_agents = []
        if sub_agent_names:
            print(f"Loading {len(sub_agent_names)} sub-agent(s)...")
            loaded_sub_agents = await self._load_sub_agents(sub_agent_names)
            print(f"Loaded {len(loaded_sub_agents)} sub-agent(s)")
        
        # Load MCP servers if specified in config
        loaded_mcp_toolsets = await self._load_mcp_servers()
        
        # Determine agent type (default to "llm" for backward compatibility)
        agent_type = self.config.get("agent_type", "llm").lower()
        agent_config = self.config.get("agent", {})
        
        if agent_type == "sequential":
            # Create SequentialAgent (workflow-oriented)
            if not loaded_sub_agents:
                raise ValueError("SequentialAgent requires at least one sub-agent in 'sub_agents' array")
            
            agent_kwargs = {
                "name": agent_config.get("name", "sequential_agent"),
                "sub_agents": loaded_sub_agents,
                "description": agent_config.get("description", "Sequential workflow agent"),
            }
            
            self.agent = SequentialAgent(**agent_kwargs)
            print(f"SequentialAgent created: {self.agent.name} (pipeline with {len(loaded_sub_agents)} steps)")
        
        else:
            # Create standard LLM Agent (default)
            agent_kwargs = {
                "name": agent_config.get("name", "agent"),
                "model": agent_config.get("model", "gemini-2.5-flash"),
                "description": agent_config.get("description", ""),
                "instruction": agent_config.get("instruction", ""),
                "tools": loaded_tools,
            }
            
            # Only add sub_agents if there are any
            if loaded_sub_agents:
                agent_kwargs["sub_agents"] = loaded_sub_agents
            
            # Add MCP toolsets if there are any
            if loaded_mcp_toolsets:
                agent_kwargs["toolsets"] = loaded_mcp_toolsets
                print(f"MCP toolsets added to agent: {len(loaded_mcp_toolsets)} toolset(s)")
            
            # Add output_schema if specified (for structured JSON output)
            # Convert JSON schema to Pydantic model
            if "response_schema" in agent_config:
                output_model = self._create_pydantic_model_from_schema(
                    agent_config["response_schema"],
                    model_name=f"{agent_config.get('name', 'agent')}_output"
                )
                agent_kwargs["output_schema"] = output_model
                print(f"Structured output enabled with Pydantic model: {output_model.__name__}")
            
            self.agent = Agent(**agent_kwargs)
            print(f"Agent created: {self.agent.name}")
        
        # Create session
        self.session = await self.session_service.create_session(
            app_name=self.app_name,
            user_id=self.user_id,
            session_id=self.session_id,
        )
        print(f"Session created")
        
        # Create runner (suppress benign app name mismatch warning)
        with suppress_adk_warnings():
            self.runner = Runner(
                agent=self.agent,
                app_name=self.app_name,
                session_service=self.session_service,
            )
        print(f"Runner created for agent '{self.agent.name}'")
    
    async def send_message(self, query: str, return_json: bool = False):
        """
        Send a message to the agent and return the response.
        
        Args:
            query: The message to send to the agent
            return_json: If True, attempts to parse response as JSON and returns dict.
                        If False or parsing fails, returns string.
            
        Returns:
            Agent's response as string, or dict if return_json=True and response is valid JSON
        """
        if not self.runner:
            raise RuntimeError("AgentManager not initialized. Call initialize() first.")
        
        print(f"Sending message: '{query}'")
        
        content = types.Content(role="user", parts=[types.Part(text=query)])
        final_response_text = "Agent did not respond to the query."
        
        async for event in self.runner.run_async(
            user_id=self.user_id,
            session_id=self.session_id,
            new_message=content
        ):
            if event.is_final_response():
                if event.content and event.content.parts:
                    final_response_text = event.content.parts[0].text
                elif event.actions and event.actions.escalate:
                    final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
                break
        
        print(f"Agent response received")
        
        # Check if agent uses output_schema (structured output)
        has_schema = "response_schema" in self.config.get("agent", {})
        
        # If output_schema is configured, parse the JSON response
        if has_schema or return_json:
            try:
                import json
                # The response should be JSON text when using output_schema
                if isinstance(final_response_text, str):
                    parsed_json = json.loads(final_response_text)
                    print(f"Structured output parsed from JSON")
                    return parsed_json
                # If it's already a dict or Pydantic model
                elif hasattr(final_response_text, 'model_dump'):
                    result_dict = final_response_text.model_dump()
                    print(f"Structured output converted to dict")
                    return result_dict
                elif hasattr(final_response_text, 'dict'):
                    result_dict = final_response_text.dict()
                    print(f"Structured output converted to dict")
                    return result_dict
                elif isinstance(final_response_text, dict):
                    print(f"Structured output already a dict")
                    return final_response_text
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse structured output as JSON: {e}")
                if has_schema:
                    # If schema was expected but parsing failed, raise error
                    print(f"Response text was: {final_response_text[:200]}")
            except Exception as e:
                print(f"Warning: Error processing structured output: {e}")
        
        return final_response_text
    
    async def get_conversation_history(self) -> list:
        """
        Retrieve the conversation history from the current session.
        
        Returns:
            List of conversation turns from the session events
        """
        if not self.session_service:
            return []
        
        try:
            session = await self.session_service.get_session(
                app_name=self.app_name,
                user_id=self.user_id,
                session_id=self.session_id
            )
            
            if not session:
                return []
            
            # Parse events from the session
            history = []
            
            if hasattr(session, 'events') and session.events:
                for event in session.events:
                    # Each event has 'content' with role and parts
                    if hasattr(event, 'content') and event.content:
                        content = event.content
                        if hasattr(content, 'role') and hasattr(content, 'parts') and content.parts:
                            # Extract text from the first part
                            text = content.parts[0].text if hasattr(content.parts[0], 'text') else str(content.parts[0])
                            
                            # Map 'model' role to 'assistant' for consistency
                            role = "assistant" if content.role == "model" else content.role
                            
                            history.append({
                                "role": role,
                                "content": text,
                                "timestamp": event.timestamp if hasattr(event, 'timestamp') else None
                            })
            
            return history
            
        except Exception as e:
            print(f"ERROR getting conversation history: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    async def close(self):
        """
        Cleanup resources including tools and MCP servers.
        """
        # Cleanup tools
        self.tool_loader.cleanup()
        
        # Stop any auto-started MCP servers
        for server in self.mcp_servers:
            try:
                server.stop_server()
            except Exception as e:
                print(f"Warning: Error stopping MCP server: {e}")
        
        print(f"AgentManager closed")


# Example usage
async def main():
    """
    Demonstrates how to use the AgentManager class.
    """
    # Create agent manager from config
    agent_manager = AgentManager("configs/01_basic_template_agent.json")
    
    # Initialize the agent, session, and runner
    await agent_manager.initialize()
    
    # Send messages and get responses
    response = await agent_manager.send_message("I would like to create a new agent for a project")
    print(f"\nResponse: {response}\n")
    
    # You can send multiple messages in the same session
    # response2 = await agent_manager.send_message("What tools should I use?")
    # print(f"\nResponse 2: {response2}\n")
    
    # Cleanup
    await agent_manager.close()


if __name__ == "__main__":
    asyncio.run(main())
