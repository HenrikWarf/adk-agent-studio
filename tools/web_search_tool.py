"""
Web Search Tool - class-based example.

This demonstrates a more complex tool that:
- Manages state (API connection)
- Uses environment variables for configuration
- Has parameter validation
- Requires setup and teardown
"""
import os
from typing import List, Optional
from agent_def.base_tool import BaseTool, ToolParameter


class WebSearchTool(BaseTool):
    """
    Search the web for information using a search API.
    
    Configuration via environment variables:
    - SEARCH_API_KEY: API key for the search service (optional for demo)
    - SEARCH_ENGINE: Search engine to use (default: "demo")
    """
    
    @property
    def name(self) -> str:
        return "web_search"
    
    @property
    def description(self) -> str:
        return "Search the web for current information on any topic"
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="string",
                description="The search query string",
                required=True
            ),
            ToolParameter(
                name="num_results",
                type="number",
                description="Number of results to return (1-10)",
                required=False,
                default=3
            )
        ]
    
    def _setup(self):
        """Setup: Load API key and configuration from environment."""
        self.api_key = os.getenv("SEARCH_API_KEY", "demo-key")
        self.search_engine = os.getenv("SEARCH_ENGINE", "demo")
        
        # In a real implementation, you might initialize an API client here
        self.initialized = True
        
        print(f"WebSearchTool initialized with engine: {self.search_engine}")
    
    async def execute(self, query: str, num_results: int = 3) -> str:
        """
        Execute a web search.
        
        Args:
            query: Search query string
            num_results: Number of results to return (default: 3)
        
        Returns:
            Formatted search results as a string
        """
        # Validate query
        if not query:
            return "Error: query parameter is required"
        
        # Limit results
        num_results = max(1, min(num_results, 10))
        
        # Demo implementation (replace with actual API call)
        if self.search_engine == "demo":
            return self._demo_search(query, num_results)
        else:
            # Real implementation would make API call here
            return await self._real_search(query, num_results)
    
    def _demo_search(self, query: str, num_results: int) -> str:
        """
        Demo search that returns mock results.
        Replace this with actual search API integration.
        """
        results = []
        
        for i in range(num_results):
            results.append(f"""
Result {i + 1}:
Title: {query} - Information and Resources
URL: https://example.com/search/{query.replace(' ', '-').lower()}-{i + 1}
Snippet: This is a demo result for the query "{query}". In a real implementation, 
this would contain actual search results from a search API.
            """.strip())
        
        formatted = f"Search results for '{query}':\n\n"
        formatted += "\n\n---\n\n".join(results)
        formatted += f"\n\nNote: These are demo results. Set SEARCH_API_KEY environment variable for real searches."
        
        return formatted
    
    async def _real_search(self, query: str, num_results: int) -> str:
        """
        Real search implementation (placeholder).
        
        In a production system, this would:
        1. Make API call to search service (Google, Bing, etc.)
        2. Parse results
        3. Format and return
        """
        try:
            # Example with a hypothetical search API:
            # response = await self.search_client.search(
            #     query=query,
            #     num_results=num_results,
            #     api_key=self.api_key
            # )
            # return self._format_results(response)
            
            return f"Real search not implemented yet. Query: {query}"
        
        except Exception as e:
            return f"Search error: {str(e)}"
    
    def _teardown(self):
        """Cleanup: Close API connections if any."""
        self.initialized = False
        print("WebSearchTool cleaned up")

