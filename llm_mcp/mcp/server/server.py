"""
Core FastMCP Server implementation.

This module contains the core server logic for the LLM-MCP framework,
focusing on FastMCP instance management and request/response handling.
"""

import logging
from typing import Optional

from fastmcp import FastMCP

from .registry import MCPToolRegistry
from ...models.semantic import SemanticSearchConfig

logger = logging.getLogger("llm-mcp-server")


class FastMCPServer:
    """
    Core FastMCP Server implementation.
    
    This server provides tools to LLMs via the Model Context Protocol using FastMCP.
    It focuses on core server functionality while delegating tool registration
    and process management to specialized components.
    """
    
    def __init__(
        self,
        server_name: str = "llm-mcp-framework",
        semantic_config: Optional[SemanticSearchConfig] = None
    ):
        """
        Initialize the FastMCP server.
        
        Args:
            server_name: Name of the server for identification
            semantic_config: Configuration for semantic search tool (optional)
        """
        self.server_name = server_name
        self.semantic_config = semantic_config
        self.mcp = FastMCP(server_name)
        self.tool_registry: Optional[MCPToolRegistry] = None
        self._initialize_server()
    
    def _initialize_server(self) -> None:
        """Initialize the server components."""
        # Create and configure tool registry
        self.tool_registry = MCPToolRegistry(self.mcp, semantic_config=self.semantic_config)
        
        # Register all framework tools
        self.tool_registry.register_framework_tools()
        
        logger.info(f"Initialized {self.server_name} with {self.tool_registry.get_tool_count()} tools")
    
    def run(self, transport: str = "stdio") -> None:
        """
        Run the FastMCP server.
        
        Args:
            transport: Transport protocol to use (default: "stdio")
        """
        logger.info(f"Starting {self.server_name} FastMCP server with {transport} transport...")
        
        try:
            self.mcp.run(transport=transport)
        except KeyboardInterrupt:
            logger.info("Server stopped by user")
        except Exception as e:
            logger.error(f"Server error: {e}")
            raise
    
    def get_server_info(self) -> dict:
        """
        Get server information.
        
        Returns:
            Dictionary with server details
        """
        tool_count = self.tool_registry.get_tool_count() if self.tool_registry else 0
        tool_names = self.tool_registry.get_tool_names() if self.tool_registry else []
        
        return {
            "server_name": self.server_name,
            "tool_count": tool_count,
            "tool_names": tool_names,
            "transport": "stdio"  # Currently only stdio is supported
        }
    
    def get_tool_registry(self) -> Optional[MCPToolRegistry]:
        """
        Get the tool registry instance.
        
        Returns:
            Tool registry instance or None
        """
        return self.tool_registry
    
    def reload_tools(self) -> bool:
        """
        Reload all tools in the registry.
        
        Returns:
            True if reload was successful
        """
        if not self.tool_registry:
            logger.error("No tool registry available for reload")
            return False
        
        try:
            # Clear existing tools
            self.tool_registry.clear_all_tools()
            
            # Re-register framework tools
            self.tool_registry.register_framework_tools()
            
            logger.info(f"Reloaded {self.tool_registry.get_tool_count()} tools")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reload tools: {e}")
            return False


# Alias for backward compatibility
MCPServer = FastMCPServer


def create_server(
    server_name: str = "llm-mcp-framework",
    semantic_config: Optional[SemanticSearchConfig] = None
) -> FastMCPServer:
    """
    Create a new FastMCP server instance.
    
    Args:
        server_name: Name of the server
        semantic_config: Configuration for semantic search tool (optional)
        
    Returns:
        FastMCP server instance
    """
    return FastMCPServer(server_name, semantic_config=semantic_config)


def run_server(
    server_name: str = "llm-mcp-framework",
    transport: str = "stdio",
    semantic_config: Optional[SemanticSearchConfig] = None
) -> None:
    """
    Create and run a FastMCP server.
    
    Args:
        server_name: Name of the server
        transport: Transport protocol to use
        semantic_config: Configuration for semantic search tool (optional)
    """
    server = create_server(server_name, semantic_config=semantic_config)
    server.run(transport)


if __name__ == "__main__":
    # Run the server directly
    run_server()