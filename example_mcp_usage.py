"""
Example usage of MCPServerManager - Creating and running MCP servers from configs.

This script demonstrates how to:
1. Load an MCP server configuration
2. Initialize the server with tools
3. Start the server
4. (Optionally) Export config for Claude Desktop
"""
import asyncio
from pathlib import Path
from mcp_def.mcp_manager import MCPServerManager


def example_1_basic_calculator_server():
    """Example 1: Load and initialize a basic calculator server."""
    print("\n" + "="*60)
    print("Example 1: Basic Calculator Server")
    print("="*60)
    
    # Create server manager
    server = MCPServerManager("configs_mcp/01_calculator_server.json")
    
    # Initialize (loads tools)
    server.initialize()
    
    # Get server info
    config = server.get_config()
    print(f"\n📊 Server Info:")
    print(f"   Name: {config['name']}")
    print(f"   Transport: {config['transport']}")
    print(f"   Tools: {', '.join([t['name'] for t in config['tools']])}")
    
    print("\n💡 To start this server, run:")
    print(f"   server.start_server()")
    print("\n⚠️  Note: This will block if using stdio transport")


def example_2_multi_tool_server():
    """Example 2: Load a server with multiple tools."""
    print("\n" + "="*60)
    print("Example 2: Multi-Tool Server")
    print("="*60)
    
    # Create server manager
    server = MCPServerManager("configs_mcp/03_multi_tool_server.json")
    
    # Initialize
    server.initialize()
    
    # Get tools
    tools = server.get_tools()
    print(f"\n🛠️  Loaded {len(tools)} tool(s):")
    for tool in tools:
        print(f"   - {tool['name']} ({tool['type']})")


def example_3_export_for_claude():
    """Example 3: Export server config for Claude Desktop."""
    print("\n" + "="*60)
    print("Example 3: Export for Claude Desktop")
    print("="*60)
    
    # Create server manager
    server = MCPServerManager("configs_mcp/01_calculator_server.json")
    
    # Export config
    claude_config = server.export_config_for_claude()
    
    print("\n📤 Claude Desktop Configuration:")
    print(json.dumps(claude_config, indent=2))
    
    print("\n💡 Add this to your Claude Desktop config file:")
    print("   Mac: ~/Library/Application Support/Claude/claude_desktop_config.json")
    print("   Windows: %APPDATA%\\Claude\\claude_desktop_config.json")


def example_4_run_server():
    """Example 4: Actually start an MCP server (blocking)."""
    print("\n" + "="*60)
    print("Example 4: Run MCP Server")
    print("="*60)
    
    # For this example, we'll use an HTTP server so it doesn't block
    # You can create a config with HTTP transport or modify an existing one
    
    print("\n⚠️  Starting a server will block the program.")
    print("   Press Ctrl+C to stop the server.")
    
    response = input("\nDo you want to start the calculator server? (y/n): ")
    
    if response.lower() == 'y':
        server = MCPServerManager("configs_mcp/01_calculator_server.json")
        server.initialize()
        
        print("\n🚀 Starting server...")
        print("   Note: stdio servers run in blocking mode")
        print("   HTTP servers run in background thread")
        
        try:
            server.start_server()
            
            # If it's a background server, keep alive
            if server.transport in ["http", "sse"]:
                print("\n✅ Server is running in background")
                print("   Press Ctrl+C to stop")
                
                try:
                    while True:
                        import time
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\n\n🛑 Stopping server...")
                    server.stop_server()
        
        except KeyboardInterrupt:
            print("\n\n🛑 Server stopped")
    else:
        print("\n  Skipped server start")


if __name__ == "__main__":
    import json
    
    print("""
╔══════════════════════════════════════════════════════════╗
║        MCP Server Manager - Example Usage                ║
║                                                            ║
║  Demonstrates how to use MCPServerManager to create      ║
║  and manage Model Context Protocol servers               ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    # Check if config files exist
    configs_dir = Path("configs_mcp")
    if not configs_dir.exists() or not list(configs_dir.glob("*.json")):
        print("❌ No MCP server configurations found in configs_mcp/")
        print("   Please create some configs first using the Streamlit UI")
        print("   Or run: streamlit run app.py")
        exit(1)
    
    # Run examples
    try:
        example_1_basic_calculator_server()
        
        input("\n\nPress Enter to continue to Example 2...")
        example_2_multi_tool_server()
        
        input("\n\nPress Enter to continue to Example 3...")
        example_3_export_for_claude()
        
        input("\n\nPress Enter to continue to Example 4...")
        example_4_run_server()
        
        print("\n\n✅ Examples completed!")
        print("\n💡 Next steps:")
        print("   1. Create more MCP servers using: streamlit run app.py")
        print("   2. Integrate servers with Claude Desktop")
        print("   3. Build custom tools in the tools/ directory")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

