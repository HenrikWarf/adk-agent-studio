"""
Example demonstrating structured JSON output with response schemas.
"""
import asyncio
import json
from agent_def import AgentManager


async def test_product_analyzer():
    """Test the product analyzer agent with structured output."""
    print("=" * 60)
    print("Example: Product Analyzer (Structured JSON Output)")
    print("=" * 60)
    
    # Create agent with response schema
    agent = AgentManager("configs/10_structured_product_analyzer_agent.json")
    await agent.initialize()
    
    # Test product description
    product_description = """
    Introducing the UltraBook Pro X1 - a premium laptop designed for 
    professionals and creative workers. Features include a stunning 15.6" 
    4K display, Intel Core i9 processor, 32GB RAM, and 1TB SSD storage. 
    The sleek aluminum body weighs just 3.5 lbs, making it perfect for 
    on-the-go productivity. Battery life up to 12 hours. Price: $2,499
    """
    
    print("\nProduct Description:")
    print(product_description)
    print("\n" + "-" * 60)
    
    # Get structured response
    response = await agent.send_message(product_description)
    
    print("\nStructured Response (JSON):")
    print(json.dumps(response, indent=2))
    
    # Access structured data
    print("\n" + "-" * 60)
    print("Accessing structured fields:")
    print(f"  Product Name: {response.get('product_name')}")
    print(f"  Category: {response.get('category')}")
    print(f"  Price Range: {response.get('price_range')}")
    print(f"  Key Features: {', '.join(response.get('key_features', []))}")
    
    await agent.close()


async def test_task_breakdown():
    """Test the task breakdown agent with structured output."""
    print("\n\n" + "=" * 60)
    print("Example: Task Breakdown (Structured JSON Output)")
    print("=" * 60)
    
    # Create agent with response schema
    agent = AgentManager("configs/11_structured_task_breakdown_agent.json")
    await agent.initialize()
    
    # Test project description
    project = "Build a web application for tracking personal finances with user authentication, dashboard, budget tracking, and expense reports"
    
    print(f"\nProject: {project}")
    print("\n" + "-" * 60)
    
    # Get structured response
    response = await agent.send_message(project)
    
    print("\nStructured Response (JSON):")
    print(json.dumps(response, indent=2))
    
    # Access structured data
    print("\n" + "-" * 60)
    print(f"Project: {response.get('project_title')}")
    print(f"Total Time: {response.get('total_estimated_time')}")
    print(f"\nTasks ({len(response.get('tasks', []))}):")
    for task in response.get('tasks', []):
        deps = f" [depends on: {task.get('dependencies')}]" if task.get('dependencies') else ""
        print(f"  {task.get('task_id')}. [{task.get('priority').upper()}] {task.get('title')} - {task.get('estimated_time')}{deps}")
    
    await agent.close()


async def main():
    """Run all examples."""
    await test_product_analyzer()
    await test_task_breakdown()
    
    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

