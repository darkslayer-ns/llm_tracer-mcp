"""
Tool registry manager for the generic LLM-MCP framework.
"""

from typing import Any, Dict, List, Optional, Type
from abc import ABC, abstractmethod

from .models.base import BaseFrameworkModel
from .models.tools import ToolSchema, ToolExecutionContext
from .models.errors import ToolError, ErrorType


class BaseTool(ABC):
    """
    Abstract base class for all MCP tools with structured responses.
    
    This is the foundation for creating type-safe, validated tools
    that return structured Pydantic models.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name (must be unique)."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for LLM understanding."""
        pass
    
    @property
    @abstractmethod
    def input_schema(self) -> Type[BaseFrameworkModel]:
        """Pydantic model for input validation."""
        pass
    
    @property
    @abstractmethod
    def output_schema(self) -> Optional[Type[BaseFrameworkModel]]:
        """Pydantic model for output validation (optional)."""
        pass
    
    @abstractmethod
    async def execute(
        self, 
        input_data: BaseFrameworkModel, 
        context: ToolExecutionContext
    ) -> BaseFrameworkModel:
        """
        Execute the tool with validated input.
        
        Args:
            input_data: Validated input data
            context: Execution context
            
        Returns:
            Structured output data
        """
        pass
    
    def validate_input(self, raw_input: Dict[str, Any]) -> BaseFrameworkModel:
        """
        Validate raw input against the input schema.
        
        Args:
            raw_input: Raw input dictionary
            
        Returns:
            Validated input model
            
        Raises:
            ValidationError: If input is invalid
        """
        try:
            return self.input_schema(**raw_input)
        except Exception as e:
            raise ValueError(f"Input validation failed for tool '{self.name}': {str(e)}")
    
    def validate_output(self, raw_output: Any) -> Optional[BaseFrameworkModel]:
        """
        Validate raw output against the output schema.
        
        Args:
            raw_output: Raw output data
            
        Returns:
            Validated output model or None if no schema
            
        Raises:
            ValidationError: If output is invalid
        """
        if self.output_schema is None:
            return raw_output
        
        try:
            if isinstance(raw_output, dict):
                return self.output_schema(**raw_output)
            elif isinstance(raw_output, self.output_schema):
                return raw_output
            else:
                # Try to convert to dict first
                if hasattr(raw_output, 'dict'):
                    return self.output_schema(**raw_output.dict())
                else:
                    return self.output_schema(data=raw_output)
        except Exception as e:
            raise ValueError(f"Output validation failed for tool '{self.name}': {str(e)}")
    
    def get_schema(self) -> ToolSchema:
        """
        Get the tool schema for registration.
        
        Returns:
            ToolSchema object
        """
        # Generate JSON schema from Pydantic model
        input_json_schema = self.input_schema.schema()
        
        output_json_schema = None
        if self.output_schema:
            output_json_schema = self.output_schema.schema()
        
        return ToolSchema(
            name=self.name,
            description=self.description,
            input_schema=input_json_schema,
            output_schema=output_json_schema
        )
    
    def __str__(self) -> str:
        return f"Tool({self.name}): {self.description}"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}')>"


class ToolRegistryManager:
    """
    Manages tool registration, discovery, and lifecycle.
    """
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.tool_schemas: Dict[str, ToolSchema] = {}
        self.is_initialized = False
    
    async def initialize(self) -> None:
        """Initialize the tool registry."""
        if self.is_initialized:
            return
        
        # Initialize any registered tools that need async setup
        for tool in self.tools.values():
            if hasattr(tool, 'initialize') and callable(getattr(tool, 'initialize')):
                await tool.initialize()
        
        self.is_initialized = True
    
    def register_tool(self, tool: BaseTool) -> None:
        """
        Register a tool instance.
        
        Args:
            tool: Tool instance to register
            
        Raises:
            ValueError: If tool name already exists
        """
        if tool.name in self.tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        
        # Validate tool
        self._validate_tool(tool)
        
        # Register tool and its schema
        self.tools[tool.name] = tool
        self.tool_schemas[tool.name] = tool.get_schema()
    
    def register_tool_schema(self, schema: ToolSchema) -> None:
        """
        Register a tool schema (for tools implemented elsewhere).
        
        Args:
            schema: Tool schema to register
        """
        if schema.name in self.tool_schemas:
            raise ValueError(f"Tool schema '{schema.name}' is already registered")
        
        self.tool_schemas[schema.name] = schema
    
    def unregister_tool(self, tool_name: str) -> bool:
        """
        Unregister a tool.
        
        Args:
            tool_name: Name of tool to unregister
            
        Returns:
            True if tool was unregistered, False if not found
        """
        removed = False
        
        if tool_name in self.tools:
            del self.tools[tool_name]
            removed = True
        
        if tool_name in self.tool_schemas:
            del self.tool_schemas[tool_name]
            removed = True
        
        return removed
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """
        Get a tool by name.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool instance or None if not found
        """
        return self.tools.get(tool_name)
    
    def get_tool_schema(self, tool_name: str) -> Optional[ToolSchema]:
        """
        Get a tool schema by name.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool schema or None if not found
        """
        return self.tool_schemas.get(tool_name)
    
    def list_tools(self) -> List[BaseTool]:
        """
        List all registered tool instances.
        
        Returns:
            List of tool instances
        """
        return list(self.tools.values())
    
    def list_tool_names(self) -> List[str]:
        """
        List all registered tool names.
        
        Returns:
            List of tool names
        """
        return list(self.tool_schemas.keys())
    
    def list_tool_schemas(self) -> List[ToolSchema]:
        """
        List all tool schemas.
        
        Returns:
            List of tool schemas
        """
        return list(self.tool_schemas.values())
    
    def has_tool(self, tool_name: str) -> bool:
        """
        Check if a tool is registered.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            True if tool is registered
        """
        return tool_name in self.tool_schemas
    
    def get_tools_by_category(self, category: str) -> List[BaseTool]:
        """
        Get tools by category (if tools have category metadata).
        
        Args:
            category: Category name
            
        Returns:
            List of tools in the category
        """
        tools = []
        for tool in self.tools.values():
            if hasattr(tool, 'category') and tool.category == category:
                tools.append(tool)
        return tools
    
    def search_tools(self, query: str) -> List[BaseTool]:
        """
        Search tools by name or description.
        
        Args:
            query: Search query
            
        Returns:
            List of matching tools
        """
        query_lower = query.lower()
        matching_tools = []
        
        for tool in self.tools.values():
            if (query_lower in tool.name.lower() or 
                query_lower in tool.description.lower()):
                matching_tools.append(tool)
        
        return matching_tools
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics.
        
        Returns:
            Dictionary with registry stats
        """
        return {
            "total_tools": len(self.tool_schemas),
            "executable_tools": len(self.tools),
            "schema_only_tools": len(self.tool_schemas) - len(self.tools),
            "is_initialized": self.is_initialized,
            "tool_names": list(self.tool_schemas.keys())
        }
    
    def _validate_tool(self, tool: BaseTool) -> None:
        """
        Validate a tool before registration.
        
        Args:
            tool: Tool to validate
            
        Raises:
            ValueError: If tool is invalid
        """
        # Check required properties
        if not tool.name or not isinstance(tool.name, str):
            raise ValueError("Tool must have a non-empty string name")
        
        if not tool.description or not isinstance(tool.description, str):
            raise ValueError("Tool must have a non-empty string description")
        
        if not tool.input_schema:
            raise ValueError("Tool must have an input_schema")
        
        # Validate that input_schema is a Pydantic model
        try:
            if not issubclass(tool.input_schema, BaseFrameworkModel):
                raise ValueError("Tool input_schema must be a Pydantic BaseFrameworkModel subclass")
        except TypeError:
            raise ValueError("Tool input_schema must be a Pydantic BaseFrameworkModel subclass")
        
        # Validate output_schema if provided
        if tool.output_schema:
            try:
                if not issubclass(tool.output_schema, BaseFrameworkModel):
                    raise ValueError("Tool output_schema must be a Pydantic BaseFrameworkModel subclass")
            except TypeError:
                raise ValueError("Tool output_schema must be a Pydantic BaseFrameworkModel subclass")
        
        # Check that execute method exists and is callable
        if not hasattr(tool, 'execute') or not callable(getattr(tool, 'execute')):
            raise ValueError("Tool must have a callable execute method")
    
    async def close(self) -> None:
        """Close the registry and clean up resources."""
        # Close any tools that need cleanup
        for tool in self.tools.values():
            if hasattr(tool, 'close') and callable(getattr(tool, 'close')):
                try:
                    await tool.close()
                except Exception:
                    pass  # Ignore cleanup errors
        
        self.tools.clear()
        self.tool_schemas.clear()
        self.is_initialized = False