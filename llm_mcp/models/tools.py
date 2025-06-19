"""
Tool-related models for the LLM-MCP framework.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union
from pydantic import Field, BaseModel

from .base import BaseFrameworkModel, TimestampedModel
from .errors import ToolError


class ToolCallStatus(str, Enum):
    """Status of a tool call."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


class ToolCall(TimestampedModel):
    """
    Represents a tool call from the LLM.
    """
    id: str = Field(..., description="Unique tool call ID")
    tool_name: str = Field(..., description="Name of the tool to call")
    arguments: Dict[str, Any] = Field(..., description="Arguments for the tool call")
    status: ToolCallStatus = Field(ToolCallStatus.PENDING, description="Current status of the tool call")
    
    def mark_running(self) -> None:
        """Mark the tool call as running."""
        self.status = ToolCallStatus.RUNNING
        self.update_timestamp()
    
    def mark_success(self) -> None:
        """Mark the tool call as successful."""
        self.status = ToolCallStatus.SUCCESS
        self.update_timestamp()
    
    def mark_error(self) -> None:
        """Mark the tool call as failed."""
        self.status = ToolCallStatus.ERROR
        self.update_timestamp()


class ToolResult(TimestampedModel):
    """
    Result from a tool execution.
    """
    tool_call_id: str = Field(..., description="ID of the tool call this result is for")
    tool_name: str = Field(..., description="Name of the tool that was executed")
    success: bool = Field(..., description="Whether the tool execution was successful")
    data: Optional[Dict[str, Any]] = Field(None, description="Structured result data")
    raw_output: Optional[str] = Field(None, description="Raw output from the tool")
    error: Optional[ToolError] = Field(None, description="Error if execution failed")
    execution_time: Optional[float] = Field(None, description="Time taken to execute (seconds)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @classmethod
    def success_result(
        cls,
        tool_call_id: str,
        tool_name: str,
        data: Optional[Dict[str, Any]] = None,
        raw_output: Optional[str] = None,
        execution_time: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "ToolResult":
        """Create a successful tool result."""
        return cls(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            success=True,
            data=data,
            raw_output=raw_output,
            execution_time=execution_time,
            metadata=metadata or {}
        )
    
    @classmethod
    def error_result(
        cls,
        tool_call_id: str,
        tool_name: str,
        error: ToolError,
        execution_time: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "ToolResult":
        """Create an error tool result."""
        return cls(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            success=False,
            error=error,
            execution_time=execution_time,
            metadata=metadata or {}
        )


class ToolSchema(BaseFrameworkModel):
    """
    Schema definition for a tool.
    """
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    input_schema: Dict[str, Any] = Field(..., description="JSON schema for input validation")
    output_schema: Optional[Dict[str, Any]] = Field(None, description="JSON schema for output validation")
    examples: List[Dict[str, Any]] = Field(default_factory=list, description="Usage examples")
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema
            }
        }
    
    def to_anthropic_format(self) -> Dict[str, Any]:
        """Convert to Anthropic tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema
        }


class ToolRegistry(BaseFrameworkModel):
    """
    Registry of available tools.
    """
    tools: Dict[str, ToolSchema] = Field(default_factory=dict, description="Registered tools by name")
    
    def register_tool(self, schema: ToolSchema) -> None:
        """Register a new tool."""
        self.tools[schema.name] = schema
    
    def get_tool(self, name: str) -> Optional[ToolSchema]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def list_tools(self) -> List[ToolSchema]:
        """List all registered tools."""
        return list(self.tools.values())
    
    def to_openai_format(self) -> List[Dict[str, Any]]:
        """Convert all tools to OpenAI format."""
        return [tool.to_openai_format() for tool in self.tools.values()]
    
    def to_anthropic_format(self) -> List[Dict[str, Any]]:
        """Convert all tools to Anthropic format."""
        return [tool.to_anthropic_format() for tool in self.tools.values()]


class ToolExecutionContext(BaseFrameworkModel):
    """
    Context for tool execution.
    """
    session_id: str = Field(..., description="Session ID")
    request_id: str = Field(..., description="Request ID")
    user_context: Dict[str, Any] = Field(default_factory=dict, description="User-provided context")
    system_context: Dict[str, Any] = Field(default_factory=dict, description="System context")
    working_directory: Optional[str] = Field(None, description="Working directory for tool execution")
    environment_vars: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    timeout: Optional[float] = Field(None, description="Execution timeout in seconds")