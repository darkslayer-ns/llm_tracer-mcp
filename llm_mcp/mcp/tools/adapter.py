"""
MCP Tool Adapter for converting framework tools to MCP format.

This module handles the adaptation of framework tools for use with MCP servers,
including parameter mapping, result formatting, and error translation.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Type

from ...models.tools import ToolExecutionContext
from ...tool_registry import BaseTool

logger = logging.getLogger("llm-mcp-tool-adapter")


class MCPToolAdapter:
    """
    Adapts framework tools for MCP usage.
    
    This class handles the conversion between framework tool formats and MCP
    tool formats, including parameter mapping, result formatting, and error handling.
    """
    
    def __init__(self):
        self.adapted_tools: Dict[str, BaseTool] = {}
    
    def register_tool(self, tool: BaseTool) -> None:
        """
        Register a framework tool for MCP adaptation.
        
        Args:
            tool: The framework tool to register
        """
        self.adapted_tools[tool.name] = tool
        logger.debug(f"Registered tool for MCP adaptation: {tool.name}")
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """
        Get a registered tool by name.
        
        Args:
            tool_name: Name of the tool to retrieve
            
        Returns:
            The tool instance or None if not found
        """
        return self.adapted_tools.get(tool_name)
    
    def get_tool_names(self) -> List[str]:
        """Get list of registered tool names."""
        return list(self.adapted_tools.keys())
    
    def get_mcp_tool_schema(self, tool: BaseTool) -> Dict[str, Any]:
        """
        Convert a framework tool to MCP tool schema format.
        
        Args:
            tool: The framework tool to convert
            
        Returns:
            MCP-compatible tool schema
        """
        # Get the input schema from the tool
        input_schema = tool.input_schema
        
        # Convert Pydantic model to JSON schema if needed
        if hasattr(input_schema, 'model_json_schema'):
            schema = input_schema.model_json_schema()
        elif hasattr(input_schema, 'schema'):
            schema = input_schema.schema()
        else:
            # Fallback for basic types
            schema = {"type": "object", "properties": {}}
        
        return {
            "name": tool.name,
            "description": tool.description,
            "inputSchema": schema
        }
    
    async def execute_tool(
        self, 
        tool_name: str, 
        arguments: Dict[str, Any],
        session_id: str = "mcp_server",
        request_id: Optional[str] = None
    ) -> str:
        """
        Execute a framework tool and return MCP-formatted result.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            session_id: Session ID for execution context
            request_id: Request ID for execution context
            
        Returns:
            JSON-formatted result string
        """
        tool = self.get_tool(tool_name)
        if not tool:
            error_result = {
                "error": f"Tool '{tool_name}' not found in adapter",
                "success": False
            }
            return json.dumps(error_result, indent=2)
        
        try:
            # Create request object from arguments
            request = tool.input_schema(**arguments)
            
            # Create execution context
            context = ToolExecutionContext(
                session_id=session_id,
                request_id=request_id or f"{tool_name}_{hash(str(arguments))}"
            )
            
            # Execute the tool
            result = await tool.execute(request, context)
            
            # Convert result to JSON for MCP
            if hasattr(result, 'model_dump'):
                return json.dumps(result.model_dump(), indent=2)
            elif hasattr(result, 'dict'):
                return json.dumps(result.dict(), indent=2)
            else:
                return json.dumps({"result": str(result)}, indent=2)
                
        except Exception as e:
            # Enhanced error logging for better debugging
            error_type = type(e).__name__
            error_msg = f"Error executing {tool_name}: {str(e)}"
            
            # Add more detailed logging for validation errors
            if "validation error" in str(e).lower():
                logger.error(f"Validation error in {tool_name}: {e}")
                logger.error(f"Arguments that caused validation error: {arguments}")
            else:
                logger.error(f"{error_type} in {tool_name}: {e}")
            
            error_result = {
                "error": error_msg,
                "error_type": error_type,
                "success": False,
                "tool_name": tool_name,
                "arguments": arguments,
                "validation_error_details": str(e) if "validation error" in str(e).lower() else None
            }
            return json.dumps(error_result, indent=2)
    
    def create_mcp_tool_function(self, tool: BaseTool):
        """
        Create an MCP tool function for a framework tool.
        
        Args:
            tool: The framework tool to create a function for
            
        Returns:
            Async function suitable for MCP registration
        """
        async def mcp_tool_function(**kwargs):
            """Dynamically created MCP tool function."""
            return await self.execute_tool(
                tool_name=tool.name,
                arguments=kwargs
            )
        
        # Set function metadata
        mcp_tool_function.__name__ = tool.name
        mcp_tool_function.__doc__ = tool.description
        
        return mcp_tool_function
    
    def get_all_mcp_schemas(self) -> List[Dict[str, Any]]:
        """
        Get MCP schemas for all registered tools.
        
        Returns:
            List of MCP-compatible tool schemas
        """
        return [
            self.get_mcp_tool_schema(tool)
            for tool in self.adapted_tools.values()
        ]
    
    def clear_tools(self) -> None:
        """Clear all registered tools."""
        self.adapted_tools.clear()
        logger.debug("Cleared all registered tools from adapter")
    
    def get_tool_count(self) -> int:
        """Get the number of registered tools."""
        return len(self.adapted_tools)