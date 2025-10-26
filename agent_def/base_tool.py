"""
Base class for creating structured tools with validation and lifecycle management.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from pydantic import BaseModel


class ToolParameter(BaseModel):
    """Defines a tool parameter with validation schema."""
    name: str
    type: str  # "string", "number", "boolean", "array", "object"
    description: str
    required: bool = True
    default: Optional[Any] = None


class BaseTool(ABC):
    """
    Abstract base class for all class-based tools.
    
    Provides:
    - Parameter validation
    - Lifecycle management (setup/teardown)
    - Schema generation for Google ADK
    - Consistent interface
    
    Tools get configuration from environment variables only.
    """
    
    def __init__(self):
        """Initialize tool and run setup."""
        self._setup()
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the tool."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what the tool does."""
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> List[ToolParameter]:
        """List of parameters the tool accepts."""
        pass
    
    def _setup(self):
        """
        Called during initialization.
        Override to set up connections, load resources from env vars, etc.
        """
        pass
    
    def _teardown(self):
        """
        Called during cleanup.
        Override to close connections, free resources, etc.
        """
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """
        Execute the tool with given parameters.
        Must be implemented by each tool.
        
        Args:
            **kwargs: Parameters as defined in self.parameters
            
        Returns:
            Tool execution result (string, dict, or any JSON-serializable type)
        """
        pass
    
    def validate_parameters(self, **kwargs) -> Dict[str, Any]:
        """
        Validate parameters before execution.
        
        Args:
            **kwargs: Parameters to validate
            
        Returns:
            Dictionary of validated parameters
            
        Raises:
            ValueError: If required parameters are missing
        """
        validated = {}
        
        for param in self.parameters:
            value = kwargs.get(param.name)
            
            if value is None:
                if param.required and param.default is None:
                    raise ValueError(f"Required parameter '{param.name}' is missing")
                value = param.default
            
            validated[param.name] = value
        
        return validated
    
    def to_schema(self) -> Dict:
        """
        Convert tool to Google ADK tool schema format.
        
        Returns:
            Dictionary representing the tool schema
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    param.name: {
                        "type": param.type,
                        "description": param.description
                    }
                    for param in self.parameters
                },
                "required": [p.name for p in self.parameters if p.required]
            }
        }
    
    def __del__(self):
        """Cleanup when tool is destroyed."""
        try:
            self._teardown()
        except Exception:
            pass  # Ignore errors during cleanup

