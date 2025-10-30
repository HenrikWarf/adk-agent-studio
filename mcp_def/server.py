"""
Template MCP Server - Standalone Example

This is a standalone example of an MCP server with hardcoded tools.
For dynamic server creation from JSON configs, use MCPServerManager from mcp_manager.py

Example usage:
    python mcp_def/server.py

For dynamic server management:
    from mcp_def.mcp_manager import MCPServerManager
    
    server = MCPServerManager("configs_mcp/01_calculator_server.json")
    server.initialize()
    server.start_server()
"""
import asyncio
import logging
import os
from typing import List, Dict, Any

from fastmcp import FastMCP

logger = logging.getLogger(__name__)
logger.basicConfig(level=logging.INFO)

# This is a standalone example - for dynamic server creation, use MCPServerManager
mcp = FastMCP("Template MCP Server for ADK Manager")

@mcp.tool
async def add(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b

@mcp.tool
async def subtract(a: int, b: int) -> int:
    """Subtract two numbers"""
    return a - b


if __name__ == "__main__":
    # Note: For production use, consider using MCPServerManager for config-based servers
    port = int(os.getenv("PORT", "8000"))
    transport = os.getenv("TRANSPORT", "stdio")  # stdio, http, or sse
    
    logger.info(f"MCP server starting with transport: {transport}")
    
    if transport == "stdio":
        # Run with stdio (default for Claude Desktop/Cursor integration)
        asyncio.run(mcp.run())
    else:
        # Run with HTTP/SSE
        logger.info(f"MCP server running on http://0.0.0.0:{port}")
        asyncio.run(
            mcp.run(
                port=port, 
                host="0.0.0.0", 
                transport=transport
            )
        )

