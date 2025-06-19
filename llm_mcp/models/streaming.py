"""
Streaming models for real-time LLM responses.
"""

from enum import Enum
from typing import Any, Dict, Optional, Union, List
from pydantic import Field

from .base import BaseFrameworkModel, TimestampedModel
from .errors import LLMError, ToolError, SessionError
from .tools import ToolCall, ToolResult


class StreamEventType(str, Enum):
    """Types of streaming events."""
    
    # Content events
    CONTENT_START = "content_start"
    CONTENT_CHUNK = "content_chunk"
    CONTENT_END = "content_end"
    
    # Tool events
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"
    TOOL_RESULT = "tool_result"
    
    # Session events
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    ITERATION_START = "iteration_start"
    ITERATION_END = "iteration_end"
    
    # Error events
    ERROR = "error"
    WARNING = "warning"
    
    # Completion events
    COMPLETE = "complete"
    PARTIAL_COMPLETE = "partial_complete"
    
    # Metadata events
    METADATA = "metadata"
    PROGRESS = "progress"


class StreamEvent(TimestampedModel):
    """
    Generic streaming event that can represent any type of real-time update.
    """
    event_type: StreamEventType = Field(..., description="Type of streaming event")
    request_id: str = Field(..., description="ID of the request this event belongs to")
    session_id: str = Field(..., description="Session ID")
    
    # Content data
    content: Optional[str] = Field(None, description="Text content for content events")
    delta: Optional[str] = Field(None, description="Incremental content change")
    
    # Tool data
    tool_call: Optional[ToolCall] = Field(None, description="Tool call information")
    tool_result: Optional[ToolResult] = Field(None, description="Tool execution result")
    
    # Error data
    error: Optional[Union[LLMError, ToolError, SessionError]] = Field(None, description="Error information")
    warning: Optional[str] = Field(None, description="Warning message")
    
    # Progress data
    progress: Optional[Dict[str, Any]] = Field(None, description="Progress information")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional event metadata")
    
    # Sequence information
    sequence_number: Optional[int] = Field(None, description="Event sequence number")
    is_final: bool = Field(False, description="Whether this is the final event")
    
    @classmethod
    def content_chunk(
        cls,
        request_id: str,
        session_id: str,
        content: str,
        delta: Optional[str] = None,
        sequence_number: Optional[int] = None,
        **kwargs
    ) -> "StreamEvent":
        """Create a content chunk event."""
        return cls(
            event_type=StreamEventType.CONTENT_CHUNK,
            request_id=request_id,
            session_id=session_id,
            content=content,
            delta=delta,
            sequence_number=sequence_number,
            **kwargs
        )
    
    @classmethod
    def tool_call_event(
        cls,
        request_id: str,
        session_id: str,
        tool_call: ToolCall,
        event_type: StreamEventType = StreamEventType.TOOL_CALL_START,
        **kwargs
    ) -> "StreamEvent":
        """Create a tool call event."""
        return cls(
            event_type=event_type,
            request_id=request_id,
            session_id=session_id,
            tool_call=tool_call,
            **kwargs
        )
    
    @classmethod
    def tool_result_event(
        cls,
        request_id: str,
        session_id: str,
        tool_result: ToolResult,
        **kwargs
    ) -> "StreamEvent":
        """Create a tool result event."""
        return cls(
            event_type=StreamEventType.TOOL_RESULT,
            request_id=request_id,
            session_id=session_id,
            tool_result=tool_result,
            **kwargs
        )
    
    @classmethod
    def error_event(
        cls,
        request_id: str,
        session_id: str,
        error: Union[LLMError, ToolError, SessionError],
        **kwargs
    ) -> "StreamEvent":
        """Create an error event."""
        return cls(
            event_type=StreamEventType.ERROR,
            request_id=request_id,
            session_id=session_id,
            error=error,
            **kwargs
        )
    
    @classmethod
    def complete_event(
        cls,
        request_id: str,
        session_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> "StreamEvent":
        """Create a completion event."""
        return cls(
            event_type=StreamEventType.COMPLETE,
            request_id=request_id,
            session_id=session_id,
            metadata=metadata or {},
            is_final=True,
            **kwargs
        )
    
    @classmethod
    def progress_event(
        cls,
        request_id: str,
        session_id: str,
        progress: Dict[str, Any],
        **kwargs
    ) -> "StreamEvent":
        """Create a progress event."""
        return cls(
            event_type=StreamEventType.PROGRESS,
            request_id=request_id,
            session_id=session_id,
            progress=progress,
            **kwargs
        )


class StreamingSession(BaseFrameworkModel):
    """
    Configuration and state for a streaming session.
    """
    session_id: str = Field(..., description="Unique session identifier")
    request_id: str = Field(..., description="Current request identifier")
    
    # Streaming configuration
    chunk_size: Optional[int] = Field(None, description="Size of content chunks")
    include_metadata: bool = Field(True, description="Whether to include metadata events")
    buffer_size: int = Field(1000, description="Event buffer size")
    
    # State tracking
    events_sent: int = Field(0, description="Number of events sent")
    bytes_sent: int = Field(0, description="Number of bytes sent")
    is_active: bool = Field(True, description="Whether session is active")
    
    # Error handling
    max_errors: int = Field(10, description="Maximum errors before terminating stream")
    error_count: int = Field(0, description="Current error count")
    
    def increment_events(self, byte_count: int = 0) -> None:
        """Increment event and byte counters."""
        self.events_sent += 1
        self.bytes_sent += byte_count
    
    def add_error(self) -> bool:
        """Add an error and return whether stream should continue."""
        self.error_count += 1
        return self.error_count < self.max_errors
    
    def terminate(self) -> None:
        """Terminate the streaming session."""
        self.is_active = False


class StreamBuffer(BaseFrameworkModel):
    """
    Buffer for managing streaming events.
    """
    events: List[StreamEvent] = Field(default_factory=list, description="Buffered events")
    max_size: int = Field(1000, description="Maximum buffer size")
    auto_flush_size: int = Field(100, description="Size at which to auto-flush")
    
    def add_event(self, event: StreamEvent) -> bool:
        """Add an event to the buffer. Returns True if buffer should be flushed."""
        self.events.append(event)
        
        # Remove old events if buffer is full
        if len(self.events) > self.max_size:
            self.events = self.events[-self.max_size:]
        
        return len(self.events) >= self.auto_flush_size
    
    def flush(self) -> List[StreamEvent]:
        """Flush and return all buffered events."""
        events = self.events.copy()
        self.events.clear()
        return events
    
    def peek(self, count: int = 10) -> List[StreamEvent]:
        """Peek at the most recent events without removing them."""
        return self.events[-count:] if self.events else []