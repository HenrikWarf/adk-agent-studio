import asyncio
import logging
import os
from typing import List, Dict, Any

from fastmcp import FastMCP

logger = logging.getLogger(__name__)
logger.basicConfig(level=logging.INFO)

mcp = FastMCP("Template MCP Server for ADK Studio")

@mcp.tool
async def add(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b

@mcp.tool
async def subtract(a: int, b: int) -> int:
    """Subtract two numbers"""
    return a - b


if __name__ == "__main__":
    port=int(os.getenv("PORT", "8000"))
    logger.info(f"MCP server is running on port {port}")
    asyncio.run(
        mcp.runasync
            (port=port, 
            host="0.0.0.0", 
            transport="http"
        )
    )

