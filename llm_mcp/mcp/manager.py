"""
Consolidated MCP Manager for high-level MCP management and orchestration.

This manager combines the best features from both existing MCPManager classes,
providing comprehensive MCP functionality including server process management,
tool execution with error handling, context management, and integration coordination.
"""

import asyncio
import uuid
from typing import Any, Dict, List, Optional

from .client import FastMCPClient
from .server.process import MCPServerProcess
from ..models.tools import ToolCall, ToolResult, ToolExecutionContext
from ..models.errors import ToolError, ErrorType
from ..tool_registry import ToolRegistryManager

import logging
logger = logging.getLogger("llm-mcp-manager")


class MCPManager:
    """
    High-level MCP management and orchestration.
    
    This class provides comprehensive MCP functionality by combining:
    - Server process lifecycle management
    - Tool execution with proper error handling
    - Context management for tool calls
    - Timeout and resource management
    - Integration coordination between components
    """
    
    def __init__(
        self,
        session_id: Optional[str] = None,
        server_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        tool_registry: Optional[ToolRegistryManager] = None,
        default_timeout: float = 30.0,
        max_concurrent_tools: int = 5,
        auto_start_server: bool = True,
        semantic_config: Optional[Any] = None
    ):
        """
        Initialize the MCP manager.
        
        Args:
            session_id: Session ID for tool execution context
            server_configs: MCP server configurations
            tool_registry: Tool registry for framework tools
            default_timeout: Default timeout for tool execution
            max_concurrent_tools: Maximum concurrent tool executions
            auto_start_server: Whether to auto-start the MCP server
            semantic_config: Semantic search configuration for MCP server
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.default_timeout = default_timeout
        self.max_concurrent_tools = max_concurrent_tools
        self.auto_start_server = auto_start_server
        self.semantic_config = semantic_config
        
        # Initialize components
        self.mcp_client = FastMCPClient(server_configs, semantic_config=semantic_config)
        self.server_process = MCPServerProcess(semantic_config=semantic_config)
        self.tool_registry = tool_registry
        
        # Execution state
        self.active_executions: Dict[str, asyncio.Task] = {}
        self.execution_semaphore = asyncio.Semaphore(max_concurrent_tools)
        self.is_initialized = False
    
    async def initialize(self) -> bool:
        """
        Initialize the MCP manager.
        
        Returns:
            True if initialization successful
        """
        if self.is_initialized:
            return True
        
        try:
            # Auto-start server if needed
            if self.auto_start_server:
                await self._ensure_server_running()
                # Give server time to fully initialize and register all tools
                
            # Initialize tool registry if provided
            if self.tool_registry:
                await self.tool_registry.initialize()
            
            # Connect to MCP servers
            success = await self.mcp_client.connect()
            if success:
                self.is_initialized = True
                logger.info(f"MCP manager initialized for session {self.session_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP manager: {e}")
            return False
    
    async def _ensure_server_running(self) -> bool:
        """
        Ensure the MCP server is running.
        
        Returns:
            True if server is running
        """
        return await self.server_process.ensure_server_running()
    
    async def execute_tool_call(
        self,
        tool_call: ToolCall,
        context: Optional[ToolExecutionContext] = None,
        timeout: Optional[float] = None
    ) -> ToolResult:
        """
        Execute a single tool call.
        
        Args:
            tool_call: The tool call to execute
            context: Execution context
            timeout: Execution timeout (uses default if not provided)
            
        Returns:
            Tool execution result
        """
        if not self.is_initialized:
            await self.initialize()
        
        # Mark tool call as running
        tool_call.mark_running()
        
        # Get timeout
        execution_timeout = timeout or (context.timeout if context else self.default_timeout)
        
        # Create execution context if not provided
        if context is None:
            context = ToolExecutionContext(
                session_id=self.session_id,
                request_id=str(uuid.uuid4()),
                timeout=execution_timeout
            )
        
        try:
            # Acquire semaphore for concurrent execution control
            async with self.execution_semaphore:
                # Use modern asyncio.timeout() for Python 3.11+
                try:
                    async with asyncio.timeout(execution_timeout):
                        # Try MCP client first
                        if self.mcp_client.is_connected:
                            result = await self.mcp_client.execute_tool_call(tool_call)
                            tool_call.mark_success()
                            return result
                        
                        # Fallback to framework tool registry if available
                        elif self.tool_registry:
                            result = await self._execute_framework_tool(tool_call, context)
                            tool_call.mark_success()
                            return result
                        
                        else:
                            tool_call.mark_error()
                            return ToolResult.error_result(
                                tool_call_id=tool_call.id,
                                tool_name=tool_call.tool_name,
                                error=ToolError(
                                    tool_name=tool_call.tool_name,
                                    error_type=ErrorType.CONNECTION_ERROR,
                                    message="No MCP connection or tool registry available",
                                    input_args=tool_call.arguments,
                                    recoverable=True
                                )
                            )
                
                except asyncio.TimeoutError:
                    tool_call.mark_error()
                    return ToolResult.error_result(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.tool_name,
                        error=ToolError(
                            tool_name=tool_call.tool_name,
                            error_type=ErrorType.TOOL_TIMEOUT,
                            message=f"Tool execution timed out after {execution_timeout} seconds",
                            input_args=tool_call.arguments,
                            recoverable=True,
                            execution_time=execution_timeout
                        ),
                        execution_time=execution_timeout
                    )
        
        except Exception as e:
            tool_call.mark_error()
            return ToolResult.error_result(
                tool_call_id=tool_call.id,
                tool_name=tool_call.tool_name,
                error=ToolError(
                    tool_name=tool_call.tool_name,
                    error_type=ErrorType.TOOL_EXECUTION_ERROR,
                    message=f"Tool execution failed: {str(e)}",
                    input_args=tool_call.arguments,
                    recoverable=True
                )
            )
    
    async def _execute_framework_tool(
        self,
        tool_call: ToolCall,
        context: ToolExecutionContext
    ) -> ToolResult:
        """
        Execute a tool using the framework tool registry.
        
        Args:
            tool_call: The tool call to execute
            context: Execution context
            
        Returns:
            Tool execution result
        """
        import time
        start_time = time.time()
        
        try:
            # Get the tool from registry
            tool = self.tool_registry.get_tool(tool_call.tool_name)
            if not tool:
                return ToolResult.error_result(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.tool_name,
                    error=ToolError(
                        tool_name=tool_call.tool_name,
                        error_type=ErrorType.TOOL_NOT_FOUND,
                        message=f"Tool '{tool_call.tool_name}' not found in registry",
                        input_args=tool_call.arguments,
                        recoverable=False
                    )
                )
            
            # Validate and execute
            validated_input = tool.validate_input(tool_call.arguments)
            result_data = await tool.execute(validated_input, context)
            
            execution_time = time.time() - start_time
            
            return ToolResult.success_result(
                tool_call_id=tool_call.id,
                tool_name=tool_call.tool_name,
                data=result_data.dict() if hasattr(result_data, 'dict') else result_data,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return ToolResult.error_result(
                tool_call_id=tool_call.id,
                tool_name=tool_call.tool_name,
                error=ToolError(
                    tool_name=tool_call.tool_name,
                    error_type=ErrorType.TOOL_EXECUTION_ERROR,
                    message=f"Framework tool execution failed: {str(e)}",
                    input_args=tool_call.arguments,
                    recoverable=True
                ),
                execution_time=execution_time
            )
    
    async def execute_tool_calls_batch(
        self,
        tool_calls: List[ToolCall],
        context: Optional[ToolExecutionContext] = None,
        parallel: bool = True
    ) -> List[ToolResult]:
        """
        Execute multiple tool calls.
        
        Args:
            tool_calls: List of tool calls to execute
            context: Execution context
            parallel: Whether to execute in parallel
            
        Returns:
            List of tool execution results
        """
        if not tool_calls:
            return []
        
        if parallel:
            # Execute in parallel
            tasks = [
                self.execute_tool_call(tool_call, context)
                for tool_call in tool_calls
            ]
            return await asyncio.gather(*tasks, return_exceptions=False)
        else:
            # Execute sequentially
            results = []
            for tool_call in tool_calls:
                result = await self.execute_tool_call(tool_call, context)
                results.append(result)
            return results
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get available tools for LLM integration."""
        return self.mcp_client.get_available_tools()
    
    def get_tool_names(self) -> List[str]:
        """Get list of available tool names."""
        return self.mcp_client.get_tool_names()
    
    def get_execution_status(self) -> Dict[str, Any]:
        """Get current execution status."""
        return {
            "session_id": self.session_id,
            "is_initialized": self.is_initialized,
            "mcp_connected": self.mcp_client.is_connected,
            "server_running": self.server_process.is_running(),
            "active_executions": len(self.active_executions),
            "max_concurrent": self.max_concurrent_tools,
            "available_slots": self.execution_semaphore._value,
            "available_tools": len(self.get_available_tools())
        }
    
    async def restart_server(self) -> bool:
        """
        Restart the MCP server.
        
        Returns:
            True if restart was successful
        """
        logger.info("Restarting MCP server")
        
        # Disconnect client
        await self.mcp_client.disconnect()
        
        # Restart server process
        success = self.server_process.restart_server_process()
        
        if success:
            # Reconnect client
            success = await self.mcp_client.connect()
        
        return success
    
    async def close(self) -> None:
        """Close the MCP manager and clean up resources."""
        logger.info(f"Closing MCP manager for session {self.session_id}")
        
        # Cancel all active executions using modern API
        for task in self.active_executions.values():
            if not task.done():
                task.cancel()  # Modern Python 3.11+ - no msg parameter
        
        # Wait for cancellations
        if self.active_executions:
            await asyncio.gather(*self.active_executions.values(), return_exceptions=True)
        
        # Disconnect MCP client
        await self.mcp_client.disconnect()
        
        # Stop server process
        self.server_process.cleanup()
        
        # Close tool registry
        if self.tool_registry:
            await self.tool_registry.close()
        
        self.is_initialized = False
        logger.info("MCP manager closed")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()