"""
Request models for the generic LLM-MCP framework.
"""

from typing import Any, Dict, List, Optional, Type
from pydantic import Field, BaseModel, validator

from .base import BaseFrameworkModel, IdentifiedModel
from .tools import ToolSchema


class GenericRequest(IdentifiedModel):
    """
    Completely generic request model that adapts to any domain.
    The framework becomes whatever you need based on the prompts and context you provide.
    """
    system_prompt: str = Field(..., description="System prompt defining the LLM's role and behavior")
    user_message: str = Field(..., description="User message or task description")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context data")
    
    # Response configuration
    response_schema: Optional[Type[BaseModel]] = Field(None, description="Optional Pydantic model to enforce response structure")
    response_format: Optional[str] = Field(None, description="Expected response format (json, text, etc.)")
    
    # LLM configuration
    model: Optional[str] = Field(None, description="Specific model to use (overrides session default)")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="LLM temperature")
    max_tokens: Optional[int] = Field(None, gt=0, description="Maximum tokens in response")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Top-p sampling parameter")
    frequency_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0, description="Presence penalty")
    
    # Execution configuration
    max_iterations: int = Field(10, gt=0, le=100, description="Maximum tool call iterations")
    timeout: Optional[float] = Field(None, gt=0, description="Request timeout in seconds")
    stream: bool = Field(False, description="Whether to stream the response")
    
    # Tool configuration
    tools_enabled: bool = Field(True, description="Whether tools are enabled for this request")
    allowed_tools: Optional[List[str]] = Field(None, description="List of allowed tool names (None = all tools)")
    tool_choice: Optional[str] = Field("auto", description="Tool choice strategy: auto, none, or specific tool name")
    
    @validator('system_prompt', 'user_message')
    def validate_non_empty_strings(cls, v):
        """Ensure prompts are not empty."""
        if not v or not v.strip():
            raise ValueError("Prompts cannot be empty")
        return v.strip()
    
    @validator('context')
    def validate_context_serializable(cls, v):
        """Ensure context is JSON serializable."""
        try:
            import json
            json.dumps(v)
            return v
        except (TypeError, ValueError) as e:
            raise ValueError(f"Context must be JSON serializable: {e}")


class BatchRequest(BaseFrameworkModel):
    """
    Request for processing multiple generic requests in batch.
    """
    requests: List[GenericRequest] = Field(..., min_items=1, description="List of requests to process")
    parallel: bool = Field(False, description="Whether to process requests in parallel")
    fail_fast: bool = Field(True, description="Whether to stop on first failure")
    batch_timeout: Optional[float] = Field(None, gt=0, description="Total timeout for batch processing")
    
    @validator('requests')
    def validate_unique_ids(cls, v):
        """Ensure all request IDs are unique."""
        ids = [req.id for req in v]
        if len(ids) != len(set(ids)):
            raise ValueError("All request IDs must be unique")
        return v


class StreamingRequest(GenericRequest):
    """
    Request specifically for streaming responses.
    """
    stream: bool = Field(True, description="Always True for streaming requests")
    chunk_size: Optional[int] = Field(None, gt=0, description="Size of streaming chunks")
    include_metadata: bool = Field(True, description="Whether to include metadata in stream events")


class ToolTestRequest(BaseFrameworkModel):
    """
    Request for testing tool functionality.
    """
    tool_name: str = Field(..., description="Name of the tool to test")
    test_inputs: List[Dict[str, Any]] = Field(..., min_items=1, description="List of test inputs")
    expected_outputs: Optional[List[Dict[str, Any]]] = Field(None, description="Expected outputs for validation")
    timeout_per_test: Optional[float] = Field(10.0, gt=0, description="Timeout per test case")


class ConfigurationRequest(BaseFrameworkModel):
    """
    Request for configuring the LLM session.
    """
    provider: Optional[str] = Field(None, description="LLM provider to use")
    model: Optional[str] = Field(None, description="Model to use")
    api_key: Optional[str] = Field(None, description="API key (if changing)")
    base_url: Optional[str] = Field(None, description="Base URL for API")
    default_temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Default temperature")
    default_max_tokens: Optional[int] = Field(None, gt=0, description="Default max tokens")
    tools: Optional[List[ToolSchema]] = Field(None, description="Tools to register")
    
    class Config:
        # Don't include sensitive fields in string representation
        repr_exclude = {"api_key"}