"""
Example of using ADK agents with MCP servers.

This script demonstrates how to:
1. Use an agent with MCP server tools (by URL)
2. Use an agent with auto-start MCP server (by config)
3. Use an agent with mixed local + MCP tools
4. Use sequential agents with MCP tools

Prerequisites:
- MCP servers must be running (for URL-based connections)
- Or use auto-start feature (requires http transport in config)
"""
import asyncio
import warnings

# Suppress deprecation warnings from Google ADK
warnings.filterwarnings('ignore', category=DeprecationWarning, module='google.adk')

from agent_def import AgentManager


async def example_1_mcp_by_url():
    """
    Example 1: Agent connecting to MCP server by URL.
    
    The MCP server must be running before starting the agent.
    """
    print("\n" + "="*70)
    print("Example 1: Agent with MCP Server (by URL)")
    print("="*70)
    print("\n💡 Make sure the MCP server is running at http://localhost:8000")
    print("   Start it with: python example_mcp_usage.py")
    
    # Create agent that connects to running MCP server
    agent = AgentManager("configs/12_mcp_calculator_agent.json")
    
    # Initialize (connects to MCP server)
    await agent.initialize()
    
    # Send a message that requires calculator tools from MCP
    query = "What is 25 * 84 + 156?"
    print(f"\n📤 Query: {query}")
    response = await agent.send_message(query)
    print(f"📥 Response: {response}")
    
    # Cleanup
    await agent.close()
    print("\n✅ Example 1 completed\n")


async def example_2_mcp_auto_start():
    """
    Example 2: Agent with auto-start MCP server.
    
    The agent will automatically start the MCP server if it's not running.
    """
    print("\n" + "="*70)
    print("Example 2: Agent with Auto-Start MCP Server")
    print("="*70)
    print("\n🚀 Agent will auto-start the MCP server...")
    
    # Create agent with auto-start enabled
    agent = AgentManager("configs/14_mcp_auto_start_agent.json")
    
    # Initialize (auto-starts MCP server in background)
    await agent.initialize()
    
    # Send a message using MCP tools
    query = "Calculate 100 divided by 5, then tell me the weather in London"
    print(f"\n📤 Query: {query}")
    response = await agent.send_message(query)
    print(f"📥 Response: {response}")
    
    # Cleanup (stops auto-started server)
    await agent.close()
    print("\n✅ Example 2 completed (MCP server stopped)\n")


async def example_3_mixed_tools():
    """
    Example 3: Agent with both local Python tools and MCP server tools.
    
    This shows how agents can seamlessly use tools from multiple sources.
    """
    print("\n" + "="*70)
    print("Example 3: Agent with Mixed Local + MCP Tools")
    print("="*70)
    print("\n🔧 Agent has access to:")
    print("   - Local Python tools: calculator")
    print("   - MCP server tools: weather (via http://localhost:8001)")
    print("\n💡 Make sure weather MCP server is running at http://localhost:8001")
    
    # Create agent with mixed tools
    agent = AgentManager("configs/13_mcp_mixed_tools_agent.json")
    
    # Initialize
    await agent.initialize()
    
    # Test local tool
    query1 = "Calculate 456 + 789 for me"
    print(f"\n📤 Query 1 (local tool): {query1}")
    response1 = await agent.send_message(query1)
    print(f"📥 Response: {response1}")
    
    # Test MCP tool
    query2 = "What's the weather like in Tokyo?"
    print(f"\n📤 Query 2 (MCP tool): {query2}")
    response2 = await agent.send_message(query2)
    print(f"📥 Response: {response2}")
    
    # Test both
    query3 = "Calculate 50 * 3 and then tell me about the weather in Paris"
    print(f"\n📤 Query 3 (both tools): {query3}")
    response3 = await agent.send_message(query3)
    print(f"📥 Response: {response3}")
    
    # Cleanup
    await agent.close()
    print("\n✅ Example 3 completed\n")


async def example_4_chat_with_mcp():
    """
    Example 4: Multi-turn conversation with MCP tools.
    
    Demonstrates that MCP tools work seamlessly in chat mode.
    """
    print("\n" + "="*70)
    print("Example 4: Chat Mode with MCP Tools")
    print("="*70)
    print("\n💬 Having a conversation using MCP tools...")
    
    # Create agent with auto-start
    agent = AgentManager("configs/14_mcp_auto_start_agent.json")
    await agent.initialize()
    
    # Multi-turn conversation
    queries = [
        "Calculate 15 + 27",
        "Now multiply that result by 3",
        "What's the weather in New York?",
        "Thanks! Can you calculate 100 divided by the first result you gave me?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n💬 Turn {i}: {query}")
        response = await agent.send_message(query)
        print(f"🤖 Agent: {response}")
    
    # Cleanup
    await agent.close()
    print("\n✅ Example 4 completed\n")


async def run_all_examples():
    """Run all examples in sequence."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║     ADK Agent + MCP Server Integration Examples                     ║
║                                                                      ║
║  Demonstrates various ways to use MCP servers with ADK agents       ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Check if user wants to run examples that require pre-running servers
    print("⚠️  Some examples require MCP servers to be running beforehand:")
    print("   - Example 1: requires calculator server at http://localhost:8000")
    print("   - Example 3: requires weather server at http://localhost:8001")
    print("\n   Examples 2 and 4 will auto-start their servers.\n")
    
    choice = input("Run all examples? (y/n, default=y): ").lower()
    if choice and choice != 'y':
        print("\nSkipping examples. To run individual examples, call them directly.")
        return
    
    try:
        # Example 2 first (auto-start, doesn't require pre-running server)
        await example_2_mcp_auto_start()
        input("\n⏸️  Press Enter to continue to next example...")
        
        # Example 4 (also auto-start)
        await example_4_chat_with_mcp()
        input("\n⏸️  Press Enter to continue to next example...")
        
        # Example 1 (requires running server)
        print("\n⚠️  Next example requires MCP server at http://localhost:8000")
        proceed = input("   Is the server running? (y/n): ").lower()
        if proceed == 'y':
            await example_1_mcp_by_url()
            input("\n⏸️  Press Enter to continue to next example...")
        else:
            print("   Skipping Example 1")
        
        # Example 3 (requires running server)
        print("\n⚠️  Next example requires MCP server at http://localhost:8001")
        proceed = input("   Is the server running? (y/n): ").lower()
        if proceed == 'y':
            await example_3_mixed_tools()
        else:
            print("   Skipping Example 3")
        
        print("\n" + "="*70)
        print("✅ All requested examples completed!")
        print("="*70)
        print("\n💡 Next steps:")
        print("   1. Try modifying the agent configs in configs/ folder")
        print("   2. Create your own MCP servers in configs_mcp/ folder")
        print("   3. Build custom tools in the tools/ directory")
        print("   4. Use the Streamlit UI for easier management: streamlit run app.py")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_all_examples())

