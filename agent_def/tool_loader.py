"""
Tool loader and manager for dynamically loading and registering tools.

Supports:
- Function-based tools (simple async functions)
- Class-based tools (inheriting from BaseTool)
- Explicit registration from config
"""
import importlib
import inspect
from typing import Any, Dict, List, Union, Callable
from pathlib import Path

from agent_def.base_tool import BaseTool


class ToolLoader:
    """
    Loads and manages tools for agents.
    
    Tools are explicitly registered via configuration and loaded
    from the tools/ directory.
    """
    
    def __init__(self, tools_dir: str = "tools"):
        """
        Initialize the tool loader.
        
        Args:
            tools_dir: Directory containing tool modules (default: "tools")
        """
        self.tools_dir = tools_dir
        self.loaded_tools: Dict[str, Union[BaseTool, Callable]] = {}
    
    def load_tools(self, tool_names: List[str]) -> List[Callable]:
        """
        Load tools by name from the tools directory.
        
        Args:
            tool_names: List of tool names to load (e.g., ["calculator", "WeatherTool"])
            
        Returns:
            List of callable functions ready for Google ADK
            
        Raises:
            ImportError: If a tool cannot be found
            ValueError: If a tool is invalid
        """
        loaded_callables = []
        
        for tool_name in tool_names:
            tool = self._load_single_tool(tool_name)
            if tool:
                self.loaded_tools[tool_name] = tool
                
                # Convert to callable for Google ADK
                if isinstance(tool, BaseTool):
                    # Wrap class-based tool's execute method
                    loaded_callables.append(tool.execute)
                else:
                    # Function-based tool is already callable
                    loaded_callables.append(tool)
        
        return loaded_callables
    
    def _load_single_tool(self, tool_name: str) -> Union[BaseTool, Callable]:
        """
        Load a single tool by name.
        
        Supports:
        1. Class-based tools: "WeatherTool" -> tools/weather_tool.py -> class WeatherTool
        2. Function-based tools: "calculator" -> tools/calculator.py -> def calculator()
        
        Args:
            tool_name: Name of the tool to load
            
        Returns:
            Tool instance or function
        """
        # Convert class name to module name (e.g., "WeatherTool" -> "weather_tool")
        module_name = self._class_name_to_module(tool_name)
        
        try:
            # Try to import from tools directory
            module = importlib.import_module(f"{self.tools_dir}.{module_name}")
            
            # Check if it's a class (first letter uppercase)
            if tool_name[0].isupper():
                # Look for class with this name
                if hasattr(module, tool_name):
                    tool_class = getattr(module, tool_name)
                    
                    # Verify it's a BaseTool subclass
                    if inspect.isclass(tool_class) and issubclass(tool_class, BaseTool):
                        # Instantiate the tool
                        tool_instance = tool_class()
                        print(f"Loaded class-based tool: {tool_name}")
                        return tool_instance
                    else:
                        raise ValueError(f"{tool_name} must inherit from BaseTool")
                else:
                    raise ImportError(f"Class {tool_name} not found in {module_name}.py")
            
            else:
                # Look for function with this name
                if hasattr(module, tool_name):
                    tool_func = getattr(module, tool_name)
                    
                    # Verify it's an async function
                    if inspect.iscoroutinefunction(tool_func):
                        print(f"Loaded function-based tool: {tool_name}")
                        return tool_func
                    else:
                        raise ValueError(f"{tool_name} must be an async function")
                else:
                    raise ImportError(f"Function {tool_name} not found in {module_name}.py")
        
        except ImportError as e:
            raise ImportError(f"Could not load tool '{tool_name}': {e}")
    
    def _class_name_to_module(self, class_name: str) -> str:
        """
        Convert a class name to module name.
        
        Examples:
            "WeatherTool" -> "weather_tool"
            "calculator" -> "calculator"
            "DatabaseQueryTool" -> "database_query_tool"
        
        Args:
            class_name: Class or function name
            
        Returns:
            Module name in snake_case
        """
        # If already lowercase, assume it's the module name
        if class_name.islower() or '_' in class_name:
            return class_name
        
        # Convert CamelCase to snake_case
        result = []
        for i, char in enumerate(class_name):
            if char.isupper() and i > 0:
                result.append('_')
            result.append(char.lower())
        
        return ''.join(result)
    
    def get_tool_schemas(self) -> List[Dict]:
        """
        Get Google ADK compatible schemas for all loaded tools.
        
        Returns:
            List of tool schema dictionaries
        """
        schemas = []
        
        for tool_name, tool in self.loaded_tools.items():
            if isinstance(tool, BaseTool):
                # Class-based tool has a to_schema method
                schemas.append(tool.to_schema())
            else:
                # Function-based tool - extract schema from function signature
                schema = self._function_to_schema(tool_name, tool)
                schemas.append(schema)
        
        return schemas
    
    def _function_to_schema(self, func_name: str, func: Callable) -> Dict:
        """
        Convert a function to a Google ADK tool schema.
        
        Extracts information from:
        - Function docstring
        - Type hints
        - Parameter names
        
        Args:
            func_name: Name of the function
            func: Function object
            
        Returns:
            Tool schema dictionary
        """
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or f"Execute {func_name}"
        
        # Extract parameters
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self' or param_name == 'cls':
                continue
            
            # Get type hint
            param_type = "string"  # Default
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int or param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
                elif param.annotation == list:
                    param_type = "array"
                elif param.annotation == dict:
                    param_type = "object"
            
            properties[param_name] = {
                "type": param_type,
                "description": f"Parameter {param_name}"
            }
            
            # Check if required (no default value)
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
        
        return {
            "name": func_name,
            "description": doc.split('\n')[0],  # First line of docstring
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    
    async def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Execute a loaded tool by name.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Parameters to pass to the tool
            
        Returns:
            Tool execution result
            
        Raises:
            ValueError: If tool not found or execution fails
        """
        if tool_name not in self.loaded_tools:
            raise ValueError(f"Tool '{tool_name}' not loaded")
        
        tool = self.loaded_tools[tool_name]
        
        if isinstance(tool, BaseTool):
            # Class-based tool with validation
            validated_params = tool.validate_parameters(**kwargs)
            return await tool.execute(**validated_params)
        else:
            # Function-based tool
            return await tool(**kwargs)
    
    def cleanup(self):
        """Cleanup all loaded tools."""
        for tool_name, tool in self.loaded_tools.items():
            if isinstance(tool, BaseTool):
                try:
                    tool._teardown()
                    print(f"Cleaned up tool: {tool_name}")
                except Exception as e:
                    print(f"Error cleaning up {tool_name}: {e}")
        
        self.loaded_tools.clear()

