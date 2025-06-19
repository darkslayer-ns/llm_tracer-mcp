"""
Pure FastMCP Client for connecting to MCP servers.

This client manages connections to MCP servers and provides tools to LLMs
for enhanced functionality. It focuses purely on connection management and
tool execution without high-level orchestration.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastmcp.client import Client
from fastmcp.utilities.mcp_config import MCPConfig, StdioMCPServer

from ..models.tools import ToolCall, ToolResult, ToolExecutionContext
from ..models.errors import ToolError, ErrorType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm-mcp-client")


class FastMCPClient:
    """
    Pure FastMCP Client for connecting to MCP servers.
    
    This client manages connections to one or more MCP servers and provides
    a unified interface for tool execution. It focuses on connection management
    and raw tool execution without process management or high-level orchestration.
    """
    
    def __init__(self, server_configs: Optional[Dict[str, Dict[str, Any]]] = None, semantic_config: Optional[Any] = None):
        """
        Initialize the FastMCP client.
        
        Args:
            server_configs: Dictionary of server configurations
                Format: {
                    "server_name": {
                        "command": "python",
                        "args": ["-m", "server.module"],
                        "env": {...}  # optional
                    }
                }
            semantic_config: Optional semantic search configuration
        """
        self.server_configs = server_configs or {}
        self.semantic_config = semantic_config
        self.client: Optional[Client] = None
        self.available_tools: List[Dict[str, Any]] = []
        self.is_connected = False
        
        # Default server configuration for the built-in server
        if not self.server_configs:
            self._setup_default_server()
    
    def _setup_default_server(self):
        """Setup default server configuration."""
        # Get the path to the server script
        server_script = Path(__file__).parent / "server" / "run_server.py"
        
        # Base arguments
        args = [str(server_script)]
        
        # Add semantic search arguments if config is provided
        if self.semantic_config and hasattr(self.semantic_config, 'qdrant_config'):
            qdrant_location = self.semantic_config.qdrant_config.location
            if qdrant_location and qdrant_location != ":memory:":
                args.extend(["--qdrant-location", qdrant_location])
                args.extend(["--collection-name", self.semantic_config.qdrant_config.collection_name])
                args.extend(["--vector-size", str(self.semantic_config.qdrant_config.vector_size)])
        
        self.server_configs = {
            "llm_mcp_framework": {
                "command": sys.executable,
                "args": args
            }
        }
    
    async def connect(self) -> bool:
        """
        Connect to MCP servers.
        
        Returns:
            True if connection successful
        """
        if self.is_connected:
            logger.info("Already connected to MCP servers")
            return True
        
        try:
            # Create MCP config
            mcp_servers = {}
            for server_name, config in self.server_configs.items():
                mcp_servers[server_name] = StdioMCPServer(
                    command=config["command"],
                    args=config.get("args", []),
                    env=config.get("env", {})
                )
            
            mcp_config = MCPConfig(mcpServers=mcp_servers)
            
            # Create and connect client
            self.client = Client(mcp_config)
            await self.client.__aenter__()
            
            # Get available tools
            tools = await self.client.list_tools()
            self.available_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                }
                for tool in tools
            ]
            
            self.is_connected = True
            logger.info(f"Connected to MCP servers with {len(self.available_tools)} tools")
            
            # Log available tools
            for tool in tools:
                logger.info(f"  - {tool.name}: {tool.description}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP servers: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from MCP servers."""
        if self.client:
            try:
                await self.client.__aexit__(None, None, None)
                logger.info("Disconnected from MCP servers")
            except Exception as e:
                logger.error(f"Error disconnecting from MCP servers: {e}")
            finally:
                self.client = None
                self.is_connected = False
                self.available_tools = []
    
    async def execute_tool_call(self, tool_call: ToolCall) -> ToolResult:
        """
        Execute a tool call via MCP.
        
        Args:
            tool_call: The tool call to execute
            
        Returns:
            Tool execution result
        """
        if not self.is_connected or not self.client:
            return ToolResult.error_result(
                tool_call_id=tool_call.id,
                tool_name=tool_call.tool_name,
                error=ToolError(
                    tool_name=tool_call.tool_name,
                    error_type=ErrorType.CONNECTION_ERROR,
                    message="Not connected to MCP server",
                    input_args=tool_call.arguments,
                    recoverable=True
                )
            )
        
        import time
        start_time = time.time()
        
        try:
            # Mark tool call as running
            tool_call.mark_running()
            
            # Execute the tool via MCP
            result = await self.client.call_tool(tool_call.tool_name, tool_call.arguments)
            
            # Extract result content
            if result and len(result) > 0:
                content = result[0].text if hasattr(result[0], 'text') else str(result[0])
                
                # Try to parse as JSON
                try:
                    result_data = json.loads(content)
                except json.JSONDecodeError:
                    result_data = {"content": content}
            else:
                result_data = {"content": "No result returned"}
            
            execution_time = time.time() - start_time
            
            # Mark as successful
            tool_call.mark_success()
            
            return ToolResult.success_result(
                tool_call_id=tool_call.id,
                tool_name=tool_call.tool_name,
                data=result_data,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            tool_call.mark_error()
            
            return ToolResult.error_result(
                tool_call_id=tool_call.id,
                tool_name=tool_call.tool_name,
                error=ToolError(
                    tool_name=tool_call.tool_name,
                    error_type=ErrorType.TOOL_EXECUTION_ERROR,
                    message=f"MCP tool execution failed: {str(e)}",
                    input_args=tool_call.arguments,
                    recoverable=True
                ),
                execution_time=execution_time
            )
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools for LLM integration."""
        return self.available_tools
    
    def get_tool_names(self) -> List[str]:
        """Get list of available tool names."""
        return [tool["function"]["name"] for tool in self.available_tools]
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()