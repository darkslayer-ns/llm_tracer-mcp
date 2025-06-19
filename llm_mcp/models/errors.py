"""
Error models for comprehensive error handling in the LLM-MCP framework.
"""

from enum import Enum
from typing import Any, Dict, Optional
from pydantic import Field

from .base import BaseFrameworkModel, TimestampedModel


class ErrorType(str, Enum):
    """Types of errors that can occur in the framework."""
    
    # LLM-related errors
    API_ERROR = "api_error"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    INVALID_RESPONSE = "invalid_response"
    MODEL_NOT_FOUND = "model_not_found"
    AUTHENTICATION_ERROR = "authentication_error"
    
    # Tool-related errors
    TOOL_NOT_FOUND = "tool_not_found"
    TOOL_EXECUTION_ERROR = "tool_execution_error"
    INVALID_TOOL_INPUT = "invalid_tool_input"
    TOOL_TIMEOUT = "tool_timeout"
    
    # Session-related errors
    SESSION_NOT_FOUND = "session_not_found"
    SESSION_EXPIRED = "session_expired"
    CONNECTION_ERROR = "connection_error"
    INITIALIZATION_ERROR = "initialization_error"
    
    # Validation errors
    VALIDATION_ERROR = "validation_error"
    SCHEMA_ERROR = "schema_error"
    
    # System errors
    SYSTEM_ERROR = "system_error"
    UNKNOWN_ERROR = "unknown_error"


class LLMError(TimestampedModel):
    """
    Comprehensive LLM-related error model.
    """
    error_type: ErrorType = Field(..., description="Type of LLM error")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    recoverable: bool = Field(True, description="Whether the error is recoverable")
    retry_after: Optional[int] = Field(None, description="Seconds to wait before retry")
    provider: Optional[str] = Field(None, description="LLM provider that caused the error")
    model: Optional[str] = Field(None, description="Model that caused the error")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    suggestion: Optional[str] = Field(None, description="Suggestion for fixing the error")
    
    def __str__(self) -> str:
        return f"LLMError({self.error_type}): {self.message}"


class ToolError(TimestampedModel):
    """
    Tool execution error model.
    """
    tool_name: str = Field(..., description="Name of the tool that failed")
    error_type: ErrorType = Field(..., description="Type of tool error")
    message: str = Field(..., description="Human-readable error message")
    input_args: Dict[str, Any] = Field(..., description="Input arguments that caused the error")
    suggestion: Optional[str] = Field(None, description="Suggestion for fixing the error")
    recoverable: bool = Field(True, description="Whether the error is recoverable")
    execution_time: Optional[float] = Field(None, description="Time spent before error (seconds)")
    
    def __str__(self) -> str:
        return f"ToolError({self.tool_name}): {self.message}"


class SessionError(TimestampedModel):
    """
    Session-level error model.
    """
    session_id: str = Field(..., description="ID of the session that encountered the error")
    error_type: ErrorType = Field(..., description="Type of session error")
    message: str = Field(..., description="Human-readable error message")
    context: Dict[str, Any] = Field(default_factory=dict, description="Session context when error occurred")
    recoverable: bool = Field(True, description="Whether the session can recover")
    requires_restart: bool = Field(False, description="Whether session needs to be restarted")
    
    def __str__(self) -> str:
        return f"SessionError({self.session_id}): {self.message}"


class ValidationError(BaseFrameworkModel):
    """
    Validation error for input/output validation.
    """
    field: str = Field(..., description="Field that failed validation")
    message: str = Field(..., description="Validation error message")
    invalid_value: Any = Field(..., description="Value that failed validation")
    expected_type: Optional[str] = Field(None, description="Expected type or format")
    
    def __str__(self) -> str:
        return f"ValidationError({self.field}): {self.message}"


class ErrorResponse(BaseFrameworkModel):
    """
    Generic error response wrapper.
    """
    success: bool = Field(False, description="Always False for error responses")
    error: LLMError | ToolError | SessionError = Field(..., description="The error that occurred")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    timestamp: str = Field(..., description="ISO timestamp when error occurred")
    
    @classmethod
    def from_error(cls, error: LLMError | ToolError | SessionError, request_id: Optional[str] = None) -> "ErrorResponse":
        """Create an ErrorResponse from any error type."""
        from datetime import datetime
        return cls(
            error=error,
            request_id=request_id,
            timestamp=datetime.utcnow().isoformat()
        )


# Exception classes that inherit from BaseException for proper exception handling
class LLMException(Exception):
    """Exception class for LLM-related errors."""
    
    def __init__(self, error: LLMError):
        self.error = error
        super().__init__(str(error))


class ToolException(Exception):
    """Exception class for tool-related errors."""
    
    def __init__(self, error: ToolError):
        self.error = error
        super().__init__(str(error))


class SessionException(Exception):
    """Exception class for session-related errors."""
    
    def __init__(self, error: SessionError):
        self.error = error
        super().__init__(str(error))