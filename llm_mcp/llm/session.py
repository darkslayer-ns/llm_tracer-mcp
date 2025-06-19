"""
Main LLMSession class - the primary entry point for the generic LLM-MCP framework.
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional, Type, Union

from ..models.base import BaseFrameworkModel
from ..models.requests import GenericRequest, BatchRequest, ConfigurationRequest
from ..models.responses import GenericResponse, BatchResponse, ResponseMetadata, ConfigurationResponse
from ..models.streaming import StreamEvent, StreamingSession, StreamEventType
from ..models.errors import LLMError, SessionError, LLMException, SessionException, ErrorType
from ..models.tools import ToolSchema, ToolRegistry
from .client import LLMClient
from ..mcp.manager import MCPManager
from ..tool_registry import ToolRegistryManager
from ..utils.logger import get_logger


class LLMSession:
    """
    Main session class for the generic LLM-MCP framework.
    
    This is the primary entry point that coordinates between:
    - LLM providers (OpenAI, Anthropic, etc.)
    - MCP tool execution
    - Request/response handling
    - Streaming capabilities
    
    The session is completely generic and adapts to any domain based on
    the prompts, tools, and context you provide.
    """
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        session_id: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize a new LLM session.
        
        Args:
            provider: LLM provider ("openai", "anthropic", etc.)
            model: Model name to use
            api_key: API key for the provider
            base_url: Base URL for API (optional)
            tools: List of tools to register
            session_id: Optional session ID (auto-generated if not provided)
            **kwargs: Additional configuration options
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.created_at = datetime.utcnow()
        self.provider = provider
        self.model = model
        
        # Initialize components
        self.llm_client: Optional[LLMClient] = None
        self.mcp_manager: Optional[MCPManager] = None
        self.tool_registry = ToolRegistryManager()
        
        # Session state
        self.is_initialized = False
        self.active_requests: Dict[str, GenericRequest] = {}
        self.streaming_sessions: Dict[str, StreamingSession] = {}
        
        # Configuration
        self.config = {
            "provider": provider,
            "model": model,
            "api_key": api_key,
            "base_url": base_url,
            **kwargs
        }
        
        # Register tools if provided
        if tools:
            for tool in tools:
                self.tool_registry.register_tool(tool)
        
        # Initialize logger
        self.logger = get_logger()
    
    async def initialize(self) -> None:
        """Initialize the session components."""
        if self.is_initialized:
            return
        
        try:
            # Initialize LLM client
            self.llm_client = LLMClient(
                provider=self.provider,
                model=self.model,
                api_key=self.config.get("api_key"),
                base_url=self.config.get("base_url"),
                **{k: v for k, v in self.config.items() if k not in ["provider", "model", "api_key", "base_url"]}
            )
            
            # Initialize MCP manager with semantic config if provided
            semantic_config = self.config.get("semantic_config")
            self.mcp_manager = MCPManager(
                session_id=self.session_id,
                semantic_config=semantic_config
            )
            
            await self.mcp_manager.initialize()
            
            self.is_initialized = True
            
        except Exception as e:
            error = SessionError(
                session_id=self.session_id,
                error_type=ErrorType.INITIALIZATION_ERROR,
                message=f"Failed to initialize session: {str(e)}",
                context={"provider": self.provider, "model": self.model},
                recoverable=False,
                requires_restart=True
            )
            raise SessionException(error)
    
    async def run_analysis(self, request: GenericRequest) -> GenericResponse:
        """
        Run a complete analysis request (non-streaming).
        
        This is the main method for processing requests. The framework
        adapts to whatever domain/task is specified in the request.
        
        Args:
            request: Generic request containing prompts, context, and configuration
            
        Returns:
            Generic response with results
        """
        await self.initialize()
        
        # Generate request ID if not provided
        if not request.id:
            request.id = str(uuid.uuid4())
        
        # Track active request
        self.active_requests[request.id] = request
        
        # Create response metadata
        metadata = ResponseMetadata(
            request_id=request.id,
            session_id=self.session_id,
            model_used=request.model or self.model,
            provider=self.provider
        )
        
        try:
            # Execute the request
            result = await self._execute_request(request, metadata)
            
            # Clean up
            self.active_requests.pop(request.id, None)
            
            return result
            
        except Exception as e:
            # Clean up on error
            self.active_requests.pop(request.id, None)
            
            # Convert to appropriate error type
            if isinstance(e, (LLMException, SessionException)):
                error = e.error  # Extract the Pydantic model from the exception
            elif isinstance(e, (LLMError, SessionError)):
                error = e
            else:
                error = LLMError(
                    error_type=ErrorType.UNKNOWN_ERROR,
                    message=str(e),
                    details={"exception_type": type(e).__name__},
                    recoverable=True
                )
            
            return GenericResponse.error_response(
                request_id=request.id,
                error=error,
                metadata=metadata
            )
    
    async def stream_analysis(self, request: GenericRequest) -> AsyncIterator[StreamEvent]:
        """
        Stream a real-time analysis request.
        
        Args:
            request: Generic request (stream will be enabled automatically)
            
        Yields:
            Stream events with real-time updates
        """
        from ..models.streaming import StreamEventType
        await self.initialize()
        
        # Ensure streaming is enabled
        request.stream = True
        
        # Generate request ID if not provided
        if not request.id:
            request.id = str(uuid.uuid4())
        
        # Create streaming session
        streaming_session = StreamingSession(
            session_id=self.session_id,
            request_id=request.id
        )
        self.streaming_sessions[request.id] = streaming_session
        
        try:
            # Send session start event
            yield StreamEvent(
                event_type=StreamEventType.SESSION_START,
                request_id=request.id,
                session_id=self.session_id,
                metadata={"model": request.model or self.model, "provider": self.provider}
            )
            
            # Execute streaming request
            async for event in self._execute_streaming_request(request, streaming_session):
                streaming_session.increment_events()
                yield event
            
            # Send completion event
            yield StreamEvent.complete_event(
                request_id=request.id,
                session_id=self.session_id,
                metadata={
                    "events_sent": streaming_session.events_sent,
                    "bytes_sent": streaming_session.bytes_sent
                }
            )
            
        except Exception as e:
            # Send error event
            error = LLMError(
                error_type=ErrorType.UNKNOWN_ERROR,
                message=str(e),
                details={"exception_type": type(e).__name__}
            ) if not isinstance(e, (LLMException, LLMError)) else (e.error if isinstance(e, LLMException) else e)
            
            yield StreamEvent.error_event(
                request_id=request.id,
                session_id=self.session_id,
                error=error
            )
        
        finally:
            # Clean up streaming session
            self.streaming_sessions.pop(request.id, None)
    
    async def stream_analysis_with_display(self, request: GenericRequest) -> None:
        """
        Stream a real-time analysis request with automatic colored display.
        
        Args:
            request: Generic request (stream will be enabled automatically)
        """
        from ..utils.logger import get_logger, ColoredFormatter
        from ..models.streaming import StreamEventType
        logger = get_logger()
        
        # Get color codes for LLM response
        llm_color = ColoredFormatter.MESSAGE_COLORS['LLM_RESPONSE']
        reset_color = ColoredFormatter.COLORS['RESET']
        
        iteration = 0
        
        # Use streaming to see real-time output with automatic display
        async for event in self.stream_analysis(request):
            if event.event_type == StreamEventType.CONTENT_CHUNK and hasattr(event, 'delta') and event.delta:
                if iteration == 0:
                    # Use logger for colored LLM response prefix
                    logger._log_with_type(20, 'LLM_RESPONSE', "🤖 LLM: ")  # 20 = INFO level
                    iteration += 1
                # Print streaming chunks with LLM response color
                print(f"{llm_color}{event.delta}{reset_color}", end="", flush=True)
            elif event.event_type == StreamEventType.TOOL_RESULT:
                iteration = 0  # Reset for next LLM response
            elif event.event_type == StreamEventType.COMPLETE:
                print("\n")  # Final newline
            elif event.event_type == StreamEventType.ERROR:
                logger.error(f"Analysis error: {event.error.message if hasattr(event, 'error') else 'Unknown error'}")
    
    async def analyze_with_response(self, request: GenericRequest, response_model: Optional[Type] = None, display: bool = True) -> 'LLMResp':
        """
        Analyze request and return structured LLMResp with parsed components.
        
        Args:
            request: Generic request for analysis
            response_model: Optional Pydantic model class to parse JSON response into
            display: Whether to show streaming output to stdout (default: True)
            
        Returns:
            LLMResp with raw response, clean text, and parsed data
        """
        from ..models.llm_response import LLMResp, JSONExtractor
        from ..models.streaming import StreamEventType
        from ..utils.logger import get_logger, ColoredFormatter
        
        # Setup display if requested
        if display:
            logger = get_logger()
            llm_color = ColoredFormatter.MESSAGE_COLORS['LLM_RESPONSE']
            reset_color = ColoredFormatter.COLORS['RESET']
            iteration = 0
        
        # Collect the complete response
        full_response = ""
        
        async for event in self.stream_analysis(request):
            if event.event_type == StreamEventType.CONTENT_CHUNK and hasattr(event, 'delta') and event.delta:
                full_response += event.delta
                
                # Display streaming output if requested
                if display:
                    if iteration == 0:
                        logger._log_with_type(20, 'LLM_RESPONSE', "🤖 LLM: ")
                        iteration += 1
                    print(f"{llm_color}{event.delta}{reset_color}", end="", flush=True)
                    
            elif event.event_type == StreamEventType.TOOL_RESULT:
                if display:
                    iteration = 0  # Reset for next LLM response
            elif event.event_type == StreamEventType.COMPLETE:
                if display:
                    print("\n")  # Final newline
            elif event.event_type == StreamEventType.ERROR:
                # Handle error case
                error_msg = event.error.message if hasattr(event, 'error') else 'Unknown error'
                if display:
                    logger.error(f"Analysis error: {error_msg}")
                return LLMResp(
                    raw_response=f"Error: {error_msg}",
                    clean_text=f"Error: {error_msg}",
                    parsed_data=None,
                    json_extracted=None,
                    parsing_success=False,
                    parsing_error=error_msg
                )
        
        # Check if the response is too short and likely indicates the LLM didn't use tools
        if len(full_response) < 200 and "tool" not in full_response.lower() and "analyze" in full_response.lower():
            # Add a note to the response about the issue
            full_response += "\n\nNOTE: The LLM response appears incomplete. This may be due to the LLM not using available tools as expected."
        
        # Parse the complete response
        return JSONExtractor.parse_response(full_response, response_model)
    
    async def process_batch(self, batch_request: BatchRequest) -> BatchResponse:
        """
        Process multiple requests in batch.
        
        Args:
            batch_request: Batch of requests to process
            
        Returns:
            Batch response with all individual results
        """
        await self.initialize()
        
        start_time = datetime.utcnow()
        responses = []
        successful = 0
        failed = 0
        
        if batch_request.parallel:
            # Process in parallel
            tasks = [self.run_analysis(req) for req in batch_request.requests]
            if batch_request.batch_timeout:
                try:
                    async with asyncio.timeout(batch_request.batch_timeout):
                        responses = await asyncio.gather(*tasks, return_exceptions=True)
                except asyncio.TimeoutError:
                    # Handle timeout by cancelling remaining tasks
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    responses = await asyncio.gather(*tasks, return_exceptions=True)
            else:
                responses = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Process sequentially
            for request in batch_request.requests:
                try:
                    response = await self.run_analysis(request)
                    responses.append(response)
                    
                    if response.success:
                        successful += 1
                    else:
                        failed += 1
                        if batch_request.fail_fast:
                            break
                            
                except Exception as e:
                    failed += 1
                    error_response = GenericResponse.error_response(
                        request_id=request.id,
                        error=LLMError(
                            error_type=ErrorType.UNKNOWN_ERROR,
                            message=str(e)
                        )
                    )
                    responses.append(error_response)
                    
                    if batch_request.fail_fast:
                        break
        
        # Count successes/failures for parallel processing
        if batch_request.parallel:
            for response in responses:
                if isinstance(response, GenericResponse) and response.success:
                    successful += 1
                else:
                    failed += 1
        
        batch_response = BatchResponse(
            success=failed == 0,
            total_requests=len(batch_request.requests),
            successful_requests=successful,
            failed_requests=failed,
            responses=responses,
            start_time=start_time
        )
        batch_response.mark_complete()
        
        return batch_response
    
    async def configure(self, config_request: ConfigurationRequest) -> ConfigurationResponse:
        """
        Update session configuration.
        
        Args:
            config_request: Configuration changes to apply
            
        Returns:
            Configuration response with applied changes
        """
        changes_applied = {}
        warnings = []
        
        try:
            # Update provider/model if specified
            if config_request.provider and config_request.provider != self.provider:
                self.provider = config_request.provider
                changes_applied["provider"] = config_request.provider
                # Will need to reinitialize
                self.is_initialized = False
            
            if config_request.model and config_request.model != self.model:
                self.model = config_request.model
                changes_applied["model"] = config_request.model
            
            # Update configuration
            for field in ["api_key", "base_url", "default_temperature", "default_max_tokens"]:
                value = getattr(config_request, field, None)
                if value is not None:
                    self.config[field] = value
                    changes_applied[field] = value
            
            # Register new tools
            if config_request.tools:
                for tool_schema in config_request.tools:
                    self.tool_registry.register_tool_schema(tool_schema)
                changes_applied["tools_registered"] = len(config_request.tools)
            
            # Reinitialize if needed
            if not self.is_initialized:
                await self.initialize()
            
            return ConfigurationResponse(
                success=True,
                changes_applied=changes_applied,
                warnings=warnings,
                current_config=self._get_current_config()
            )
            
        except Exception as e:
            return ConfigurationResponse(
                success=False,
                changes_applied=changes_applied,
                warnings=[f"Configuration failed: {str(e)}"],
                current_config=self._get_current_config()
            )
    
    async def _execute_request(self, request: GenericRequest, metadata: ResponseMetadata) -> GenericResponse:
        """Execute a non-streaming request."""
        if not self.llm_client:
            error = SessionError(
                session_id=self.session_id,
                error_type=ErrorType.INITIALIZATION_ERROR,
                message="LLM client not initialized",
                recoverable=False
            )
            raise SessionException(error)
        
        # Get available tools from MCP manager
        mcp_tool_schemas = []
        if request.tools_enabled and self.mcp_manager:
            mcp_tools = self.mcp_manager.get_available_tools()
            # Convert MCP tools to ToolSchema objects
            from ..models.tools import ToolSchema
            for tool in mcp_tools:
                if "function" in tool:
                    func = tool["function"]
                    mcp_tool_schemas.append(ToolSchema(
                        name=func["name"],
                        description=func["description"],
                        input_schema=func.get("parameters", {})
                    ))
        
        # Get tools from tool registry
        registry_tools = self.tool_registry.list_tool_schemas() if request.tools_enabled else []
        
        # Combine MCP tools and registry tools
        available_tools = mcp_tool_schemas + registry_tools
        
        # Filter tools if specific tools are allowed
        if request.allowed_tools:
            available_tools = [
                tool for tool in available_tools
                if tool.name in request.allowed_tools
            ]
        
        # Execute the request with LLM client
        response = await self.llm_client.complete(request, available_tools)
        
        # Update metadata
        response.metadata.session_id = self.session_id
        response.metadata.tools_used = [tool.name for tool in available_tools]
        
        return response
    
    async def _execute_streaming_request(
        self,
        request: GenericRequest,
        streaming_session: StreamingSession
    ) -> AsyncIterator[StreamEvent]:
        """Execute a streaming request with tool execution loop."""
        if not self.llm_client:
            yield StreamEvent.error_event(
                request_id=request.id,
                session_id=self.session_id,
                error=SessionError(
                    session_id=self.session_id,
                    error_type=ErrorType.INITIALIZATION_ERROR,
                    message="LLM client not initialized",
                    recoverable=False
                )
            )
            return
        
        # Get available tools from MCP manager
        mcp_tool_schemas = []
        if request.tools_enabled and self.mcp_manager:
            mcp_tools = self.mcp_manager.get_available_tools()
            # Convert MCP tools to ToolSchema objects
            from ..models.tools import ToolSchema
            for tool in mcp_tools:
                if "function" in tool:
                    func = tool["function"]
                    mcp_tool_schemas.append(ToolSchema(
                        name=func["name"],
                        description=func["description"],
                        input_schema=func.get("parameters", {})
                    ))
        
        # Get tools from tool registry
        registry_tools = self.tool_registry.list_tool_schemas() if request.tools_enabled else []
        
        # Combine MCP tools and registry tools
        available_tools = mcp_tool_schemas + registry_tools
        
        # Filter tools if specific tools are allowed
        if request.allowed_tools:
            available_tools = [
                tool for tool in available_tools
                if tool.name in request.allowed_tools
            ]
        
        # Log available tools
        if available_tools:
            self.logger.available_tools(available_tools)
        
        # Build conversation messages
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.user_message})
        
        max_iterations = 50
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Create a new request for this iteration
            iter_request = GenericRequest(
                system_prompt=request.system_prompt if iteration == 1 else "Continue the conversation",
                user_message=request.user_message if iteration == 1 else "Please continue based on the tool results",
                messages=messages if iteration > 1 else None,
                tools_enabled=request.tools_enabled,
                stream=True
            )
            
            # Log TX to LLM
            if iteration == 1:
                self.logger.tx_llm(f"System: {iter_request.system_prompt[:100]}...")
                self.logger.tx_llm(f"User: {iter_request.user_message}")
            else:
                self.logger.tx_llm("Continuing conversation with tool results")
            
            # Stream the LLM response
            tool_calls = []
            content_chunks = []
            
            async for event in self.llm_client.stream_complete(iter_request, available_tools):
                event.session_id = self.session_id
                
                if event.event_type == StreamEventType.CONTENT_CHUNK:
                    content_chunks.append(event.delta or "")
                    yield event
                elif event.event_type == StreamEventType.TOOL_CALL_START:
                    tool_calls.append(event.tool_call)
                    yield event
                else:
                    yield event
            
            # If no tool calls, we're done
            if not tool_calls:
                break
            
            # Execute tool calls
            assistant_content = "".join(content_chunks)
            # Convert tool calls to dict format for OpenAI API
            tool_calls_dict = []
            for tc in tool_calls:
                if hasattr(tc, 'model_dump'):
                    tc_dict = tc.model_dump()
                else:
                    tc_dict = tc.dict()
                
                # Ensure OpenAI-compatible format
                openai_tool_call = {
                    "id": tc_dict["id"],
                    "type": "function",
                    "function": {
                        "name": tc_dict["tool_name"],
                        "arguments": json.dumps(tc_dict["arguments"])
                    }
                }
                tool_calls_dict.append(openai_tool_call)
            
            if assistant_content:
                messages.append({"role": "assistant", "content": assistant_content, "tool_calls": tool_calls_dict})
            else:
                messages.append({"role": "assistant", "tool_calls": tool_calls_dict})
            
            # Execute each tool call
            for tool_call in tool_calls:
                # Log tool call
                self.logger.tool_call(tool_call.tool_name, tool_call.arguments)
                
                yield StreamEvent(
                    event_type=StreamEventType.TOOL_CALL_START,
                    request_id=request.id,
                    session_id=self.session_id,
                    tool_call=tool_call
                )
                
                # Execute via MCP manager
                if self.mcp_manager:
                    result = await self.mcp_manager.execute_tool_call(tool_call)
                    
                    # Log tool result
                    result_preview = str(result.data)[:100] if result.success and result.data else None
                    self.logger.tool_result(tool_call.tool_name, result.success, result_preview)
                    
                    yield StreamEvent(
                        event_type=StreamEventType.TOOL_RESULT,
                        request_id=request.id,
                        session_id=self.session_id,
                        tool_result=result
                    )
                    
                    # Add tool result to conversation
                    if result.success:
                        # Handle JSON serialization of tool result data
                        try:
                            content = json.dumps(result.data, default=str)
                        except (TypeError, ValueError):
                            # Fallback to string representation if JSON serialization fails
                            content = str(result.data)
                    else:
                        content = f"Error: {result.error.message}"
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": content
                    })
                else:
                    # No MCP manager, add error
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": "Error: MCP manager not available"
                    })
        
        # If we hit max iterations, send a warning
        if iteration >= max_iterations:
            yield StreamEvent(
                event_type=StreamEventType.ERROR,
                request_id=request.id,
                session_id=self.session_id,
                error=LLMError(
                    error_type=ErrorType.MAX_ITERATIONS_REACHED,
                    message=f"Reached maximum iterations ({max_iterations})",
                    recoverable=True
                )
            )
    
    def _get_current_config(self) -> Dict[str, Any]:
        """Get current configuration (excluding sensitive data)."""
        config = self.config.copy()
        # Remove sensitive information
        config.pop("api_key", None)
        config.update({
            "session_id": self.session_id,
            "provider": self.provider,
            "model": self.model,
            "is_initialized": self.is_initialized,
            "tools_registered": len(self.tool_registry.list_tools())
        })
        return config
    
    async def close(self) -> None:
        """Close the session and clean up resources."""
        if self.mcp_manager:
            await self.mcp_manager.close()
        
        if self.llm_client:
            await self.llm_client.close()
        
        self.active_requests.clear()
        self.streaming_sessions.clear()
        self.is_initialized = False
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()