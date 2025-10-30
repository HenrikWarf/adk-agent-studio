"""
MCP Server Manager - Dynamic MCP server creation from JSON configurations.

This module provides a class-based approach to creating and managing FastMCP
servers with dynamically loaded tools from the tools/ folder.
"""
import json
import os
import sys
import asyncio
import importlib.util
import inspect
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

from fastmcp import FastMCP

# Load environment variables
load_dotenv()


class MCPServerManager:
    """
    Manages the full lifecycle of an MCP server including configuration loading,
    tool registration, and server execution.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the MCPServerManager with a configuration file.
        
        Args:
            config_path: Path to the JSON configuration file
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        # Extract server configuration
        server_config = self.config.get("server", {})
        self.name = server_config.get("name", "MCP Server")
        self.description = server_config.get("description", "")
        self.port = server_config.get("port", 8000)
        self.host = server_config.get("host", "0.0.0.0")
        self.transport = server_config.get("transport", "stdio")
        
        # Initialize FastMCP instance
        self.mcp = FastMCP(self.name)
        
        # Server state
        self.server_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.tools_loaded = []
        
        print(f"MCPServerManager created: {self.name} (transport: {self.transport})")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file."""
        path = Path(config_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(path, 'r') as f:
            return json.load(f)
    
    def _load_tool_from_file(self, tool_config: Dict[str, Any]) -> None:
        """
        Dynamically load and register a tool from a Python file.
        
        Args:
            tool_config: Tool configuration dictionary with 'file', 'type', and function/class name
        """
        tool_file = tool_config.get("file")
        tool_type = tool_config.get("type")  # 'function' or 'class'
        
        if not tool_file:
            raise ValueError("Tool configuration must include 'file' field")
        
        # Construct full path to tool file
        tools_dir = Path(__file__).parent.parent / "tools"
        tool_path = tools_dir / tool_file
        
        if not tool_path.exists():
            raise FileNotFoundError(f"Tool file not found: {tool_path}")
        
        # Load module dynamically
        spec = importlib.util.spec_from_file_location(tool_path.stem, tool_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Register tool based on type
        if tool_type == "function":
            function_name = tool_config.get("function_name")
            if not function_name:
                raise ValueError("Function-based tool must specify 'function_name'")
            
            if not hasattr(module, function_name):
                raise AttributeError(f"Function '{function_name}' not found in {tool_file}")
            
            func = getattr(module, function_name)
            
            # Register the function as an MCP tool
            self.mcp.tool(func)
            self.tools_loaded.append({
                "name": function_name,
                "type": "function",
                "file": tool_file
            })
            print(f"  ✓ Registered function tool: {function_name}")
        
        elif tool_type == "class":
            class_name = tool_config.get("class_name")
            if not class_name:
                raise ValueError("Class-based tool must specify 'class_name'")
            
            if not hasattr(module, class_name):
                raise AttributeError(f"Class '{class_name}' not found in {tool_file}")
            
            tool_class = getattr(module, class_name)
            
            # Instantiate the class
            tool_instance = tool_class()
            
            # Call setup if it exists
            if hasattr(tool_instance, '_setup'):
                tool_instance._setup()
            
            # Get the execute method
            if not hasattr(tool_instance, 'execute'):
                raise AttributeError(f"Class '{class_name}' must have an 'execute' method")
            
            execute_method = tool_instance.execute
            
            # Get tool metadata
            tool_name = tool_instance.name if hasattr(tool_instance, 'name') else class_name.lower()
            tool_description = tool_instance.description if hasattr(tool_instance, 'description') else ""
            
            # Create a wrapper function with proper signature
            # We need to inspect the execute method to get its parameters
            sig = inspect.signature(execute_method)
            params = list(sig.parameters.values())
            
            # Create the wrapper function dynamically
            async def tool_wrapper(*args, **kwargs):
                return await execute_method(*args, **kwargs)
            
            # Set function metadata
            tool_wrapper.__name__ = tool_name
            tool_wrapper.__doc__ = tool_description
            tool_wrapper.__signature__ = sig
            
            # Register the wrapped method as an MCP tool
            self.mcp.tool(tool_wrapper)
            self.tools_loaded.append({
                "name": tool_name,
                "type": "class",
                "file": tool_file,
                "class_name": class_name
            })
            print(f"  ✓ Registered class tool: {tool_name} ({class_name})")
        
        else:
            raise ValueError(f"Unknown tool type: {tool_type}. Must be 'function' or 'class'")
    
    def initialize(self):
        """
        Initialize the MCP server by loading and registering all configured tools.
        """
        print(f"\n🔧 Initializing MCP Server: {self.name}")
        print(f"   Description: {self.description}")
        print(f"   Transport: {self.transport}")
        
        if self.transport in ["http", "sse"]:
            print(f"   Host: {self.host}")
            print(f"   Port: {self.port}")
        
        # Load tools
        tools_config = self.config.get("tools", [])
        print(f"\n📦 Loading {len(tools_config)} tool(s)...")
        
        for tool_config in tools_config:
            try:
                self._load_tool_from_file(tool_config)
            except Exception as e:
                print(f"  ✗ Failed to load tool from {tool_config.get('file', 'unknown')}: {e}")
                raise
        
        print(f"\n✅ MCP Server initialized successfully with {len(self.tools_loaded)} tool(s)")
        return self
    
    def start_server(self):
        """
        Start the MCP server in a background thread (for HTTP/SSE transport)
        or run it directly (for stdio transport).
        """
        if self.is_running:
            print("⚠️  Server is already running")
            return
        
        print(f"\n🚀 Starting MCP Server: {self.name}")
        
        if self.transport == "stdio":
            # For stdio, we run synchronously
            print("   Running with stdio transport (blocking mode)")
            self.is_running = True
            asyncio.run(self.mcp.run())
        
        elif self.transport in ["http", "sse"]:
            # For HTTP/SSE, we run in a background thread
            print(f"   Starting server at http://{self.host}:{self.port}")
            
            def run_server():
                asyncio.run(
                    self.mcp.run(
                        port=self.port,
                        host=self.host,
                        transport=self.transport
                    )
                )
            
            self.server_thread = threading.Thread(target=run_server, daemon=True)
            self.server_thread.start()
            self.is_running = True
            print(f"   ✓ Server started in background thread")
        
        else:
            raise ValueError(f"Unsupported transport: {self.transport}")
    
    def stop_server(self):
        """
        Stop the MCP server.
        """
        if not self.is_running:
            print("⚠️  Server is not running")
            return
        
        print(f"\n🛑 Stopping MCP Server: {self.name}")
        
        # Note: FastMCP doesn't provide a clean shutdown method yet
        # For now, we just mark it as not running
        self.is_running = False
        
        if self.server_thread:
            print("   Note: Background thread may take a moment to terminate")
        
        print("   ✓ Server stopped")
    
    def get_config(self) -> dict:
        """
        Get the current server configuration.
        
        Returns:
            Configuration dictionary
        """
        return {
            "name": self.name,
            "description": self.description,
            "transport": self.transport,
            "host": self.host,
            "port": self.port,
            "tools": self.tools_loaded,
            "is_running": self.is_running
        }
    
    def get_tools(self) -> List[Dict[str, str]]:
        """
        Get the list of loaded tools.
        
        Returns:
            List of tool information dictionaries
        """
        return self.tools_loaded
    
    def export_config_for_claude(self) -> dict:
        """
        Export server configuration in a format suitable for Claude Desktop/Cursor.
        
        Returns:
            Configuration dictionary for Claude Desktop
        """
        # For Claude Desktop, we need to provide the command to run the server
        # This assumes the server will be run from the project root
        server_script = f"python -m mcp_def.server_{Path(self.config_path).stem}"
        
        return {
            "mcpServers": {
                self.name: {
                    "command": "python",
                    "args": ["-m", f"mcp_def.server_{Path(self.config_path).stem}"],
                    "description": self.description
                }
            }
        }

