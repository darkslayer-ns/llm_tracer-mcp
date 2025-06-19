"""
Response models for the generic LLM-MCP framework.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pydantic import Field, BaseModel

from .base import BaseFrameworkModel, TimestampedModel
from .errors import LLMError, ToolError, SessionError
from .tools import ToolCall, ToolResult


class ResponseMetadata(BaseFrameworkModel):
    """
    Metadata about the response generation.
    """
    request_id: str = Field(..., description="ID of the request this response is for")
    session_id: str = Field(..., description="Session ID")
    model_used: str = Field(..., description="Model that generated the response")
    provider: str = Field(..., description="LLM provider used")
    
    # Token usage
    prompt_tokens: Optional[int] = Field(None, description="Tokens used in prompt")
    completion_tokens: Optional[int] = Field(None, description="Tokens used in completion")
    total_tokens: Optional[int] = Field(None, description="Total tokens used")
    
    # Timing
    start_time: datetime = Field(default_factory=datetime.utcnow, description="When processing started")
    end_time: Optional[datetime] = Field(None, description="When processing completed")
    total_duration: Optional[float] = Field(None, description="Total processing time in seconds")
    
    # Tool usage
    tool_calls_made: int = Field(0, description="Number of tool calls made")
    tools_used: List[str] = Field(default_factory=list, description="Names of tools used")
    
    # Iterations
    iterations_completed: int = Field(0, description="Number of iterations completed")
    max_iterations_reached: bool = Field(False, description="Whether max iterations was reached")
    
    def mark_complete(self) -> None:
        """Mark the response as complete and calculate duration."""
        self.end_time = datetime.utcnow()
        if self.start_time:
            self.total_duration = (self.end_time - self.start_time).total_seconds()


class GenericResponse(TimestampedModel):
    """
    Completely generic response model that can contain any structured data.
    """
    success: bool = Field(..., description="Whether the request was successful")
    request_id: str = Field(..., description="ID of the request this response is for")
    
    # Response content
    result: Optional[Dict[str, Any]] = Field(None, description="Generic result data")
    parsed_result: Optional[BaseModel] = Field(None, description="Parsed result if response_schema was provided")
    raw_content: Optional[str] = Field(None, description="Raw LLM response content")
    
    # Error handling
    error: Optional[Union[LLMError, ToolError, SessionError]] = Field(None, description="Error if request failed")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings")
    
    # Execution details
    iterations: int = Field(0, description="Number of iterations completed")
    tool_calls: List[ToolCall] = Field(default_factory=list, description="Tool calls made during execution")
    tool_results: List[ToolResult] = Field(default_factory=list, description="Results from tool executions")
    
    # Metadata
    metadata: ResponseMetadata = Field(..., description="Response metadata")
    
    @classmethod
    def success_response(
        cls,
        request_id: str,
        result: Optional[Dict[str, Any]] = None,
        raw_content: Optional[str] = None,
        parsed_result: Optional[BaseModel] = None,
        metadata: Optional[ResponseMetadata] = None,
        **kwargs
    ) -> "GenericResponse":
        """Create a successful response."""
        if metadata:
            metadata.mark_complete()
        
        return cls(
            success=True,
            request_id=request_id,
            result=result,
            raw_content=raw_content,
            parsed_result=parsed_result,
            metadata=metadata or ResponseMetadata(request_id=request_id, session_id="", model_used="", provider=""),
            **kwargs
        )
    
    @classmethod
    def error_response(
        cls,
        request_id: str,
        error: Union[LLMError, ToolError, SessionError],
        metadata: Optional[ResponseMetadata] = None,
        **kwargs
    ) -> "GenericResponse":
        """Create an error response."""
        if metadata:
            metadata.mark_complete()
        
        return cls(
            success=False,
            request_id=request_id,
            error=error,
            metadata=metadata or ResponseMetadata(request_id=request_id, session_id="", model_used="", provider=""),
            **kwargs
        )


class BatchResponse(BaseFrameworkModel):
    """
    Response for batch processing.
    """
    success: bool = Field(..., description="Whether the entire batch was successful")
    total_requests: int = Field(..., description="Total number of requests in batch")
    successful_requests: int = Field(..., description="Number of successful requests")
    failed_requests: int = Field(..., description="Number of failed requests")
    
    responses: List[GenericResponse] = Field(..., description="Individual responses")
    batch_metadata: Dict[str, Any] = Field(default_factory=dict, description="Batch-level metadata")
    
    # Timing
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = Field(None)
    total_duration: Optional[float] = Field(None)
    
    def mark_complete(self) -> None:
        """Mark the batch as complete."""
        self.end_time = datetime.utcnow()
        self.total_duration = (self.end_time - self.start_time).total_seconds()


class ToolTestResponse(BaseFrameworkModel):
    """
    Response for tool testing.
    """
    tool_name: str = Field(..., description="Name of the tested tool")
    total_tests: int = Field(..., description="Total number of tests run")
    passed_tests: int = Field(..., description="Number of tests that passed")
    failed_tests: int = Field(..., description="Number of tests that failed")
    
    test_results: List[Dict[str, Any]] = Field(..., description="Individual test results")
    overall_success: bool = Field(..., description="Whether all tests passed")
    
    execution_time: float = Field(..., description="Total execution time for all tests")


class ConfigurationResponse(BaseFrameworkModel):
    """
    Response for configuration changes.
    """
    success: bool = Field(..., description="Whether configuration was successful")
    changes_applied: Dict[str, Any] = Field(..., description="Configuration changes that were applied")
    warnings: List[str] = Field(default_factory=list, description="Configuration warnings")
    current_config: Dict[str, Any] = Field(..., description="Current configuration state")