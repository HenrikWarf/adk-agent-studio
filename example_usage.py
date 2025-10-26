"""
Example usage of the AgentManager class.
Demonstrates how to create and use multiple different agents from config files.
"""
import asyncio
from agent_def import AgentManager


async def example_single_agent():
    """Example: Using a single agent for a conversation."""
    print("=" * 60)
    print("Example 1: Single Agent Conversation")
    print("=" * 60)
    
    # Create and initialize agent
    agent = AgentManager("configs/01_basic_template_agent.json")
    await agent.initialize()
    
    # Send a message
    response = await agent.send_message("Hello! Can you help me?")
    print(f"\nAgent says: {response}\n")
    
    # Cleanup
    await agent.close()


async def example_multiple_agents():
    """Example: Using multiple different agents."""
    print("=" * 60)
    print("Example 2: Multiple Agents")
    print("=" * 60)
    
    # Create two different agents
    template_agent = AgentManager("configs/01_basic_template_agent.json")
    code_reviewer = AgentManager("configs/08_sequential_code_reviewer_agent.json")
    
    # Initialize both
    await template_agent.initialize()
    await code_reviewer.initialize()
    
    # Use template agent
    print("\n--- Template Agent ---")
    response1 = await template_agent.send_message("I need a data analysis agent")
    print(f"Template Agent: {response1}\n")
    
    # Use code reviewer agent
    print("\n--- Code Reviewer Agent ---")
    code_sample = """
def calculate_sum(numbers):
    sum = 0
    for i in range(len(numbers)):
        sum = sum + numbers[i]
    return sum
"""
    response2 = await code_reviewer.send_message(f"Please review this code:\n{code_sample}")
    print(f"Code Reviewer: {response2}\n")
    
    # Cleanup
    await template_agent.close()
    await code_reviewer.close()


async def example_conversation_flow():
    """Example: Multi-turn conversation with an agent."""
    print("=" * 60)
    print("Example 3: Multi-turn Conversation")
    print("=" * 60)
    
    agent = AgentManager("configs/01_basic_template_agent.json")
    await agent.initialize()
    
    # Multi-turn conversation
    messages = [
        "I want to create a customer support agent",
        "What tools should it have?",
        "How should I handle escalations?"
    ]
    
    for msg in messages:
        print(f"\nUser: {msg}")
        response = await agent.send_message(msg)
        print(f"Agent: {response}")
    
    await agent.close()


async def main():
    """Run all examples."""
    # Run example 1
    await example_single_agent()
    
    print("\n" * 2)
    
    # Run example 2
    await example_multiple_agents()
    
    # Uncomment to run example 3
    # print("\n" * 2)
    # await example_conversation_flow()


if __name__ == "__main__":
    asyncio.run(main())

