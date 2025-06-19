"""
Generic LLM client with streaming support for multiple providers.
"""

import asyncio
import json
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional, Union
from abc import ABC, abstractmethod


def json_serializer(obj):
    """Custom JSON serializer that handles datetime objects."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

from ..models.requests import GenericRequest
from ..models.responses import GenericResponse, ResponseMetadata
from ..models.streaming import StreamEvent, StreamEventType
from ..models.errors import LLMError, LLMException, ErrorType
from ..models.tools import ToolCall, ToolResult, ToolSchema


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        self.api_key = api_key
        self.base_url = base_url
        self.config = kwargs
    
    @abstractmethod
    async def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Complete a non-streaming request."""
        pass
    
    @abstractmethod
    async def stream_complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """Complete a streaming request."""
        pass
    
    @abstractmethod
    def format_tools(self, tools: List[ToolSchema]) -> List[Dict[str, Any]]:
        """Format tools for this provider's API."""
        pass
    
    @abstractmethod
    def parse_tool_calls(self, response: Dict[str, Any]) -> List[ToolCall]:
        """Parse tool calls from provider response."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider."""
    
    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(api_key, base_url, **kwargs)
        self.client = None
    
    async def _get_client(self):
        """Get or create OpenAI client."""
        if self.client is None:
            try:
                import openai
                import httpx
                
                # Configure httpx client
                http_client = httpx.AsyncClient(
                    verify=self.config.get("verify_ssl", True),
                    timeout=httpx.Timeout(self.config.get("timeout", 60.0))
                )
                
                self.client = openai.AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    http_client=http_client
                )
            except ImportError:
                error = LLMError(
                    error_type=ErrorType.SYSTEM_ERROR,
                    message="OpenAI package not installed. Install with: pip install openai",
                    recoverable=False
                )
                raise LLMException(error)
                raise LLMException(error)
        
        return self.client
    
    async def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Complete a non-streaming OpenAI request."""
        client = await self._get_client()
        
        try:
            params = {
                "model": kwargs.get("model", "gpt-4"),
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens"),
                "top_p": kwargs.get("top_p"),
                "frequency_penalty": kwargs.get("frequency_penalty"),
                "presence_penalty": kwargs.get("presence_penalty"),
            }
            
            # Add tools if provided
            if tools:
                params["tools"] = tools
                params["tool_choice"] = kwargs.get("tool_choice", "auto")
            
            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}
            
            response = await client.chat.completions.create(**params)
            return response.model_dump()
            
        except Exception as e:
            error = LLMError(
                error_type=ErrorType.API_ERROR,
                message=f"OpenAI API error: {str(e)}",
                details={"provider": "openai", "model": kwargs.get("model", "gpt-4")},
                recoverable=True
            )
            raise LLMException(error)
            raise LLMException(error)
    
    async def stream_complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """Complete a streaming OpenAI request."""
        client = await self._get_client()
        
        try:
            params = {
                "model": kwargs.get("model", "gpt-4"),
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens"),
                "stream": True,
            }
            
            if tools:
                params["tools"] = tools
                params["tool_choice"] = kwargs.get("tool_choice", "auto")
            
            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}
            
            async for chunk in await client.chat.completions.create(**params):
                yield chunk.model_dump()
                
        except Exception as e:
            error = LLMError(
                error_type=ErrorType.API_ERROR,
                message=f"OpenAI streaming error: {str(e)}",
                details={"provider": "openai", "model": kwargs.get("model", "gpt-4")},
                recoverable=True
            )
            raise LLMException(error)
            raise LLMException(error)
    
    def format_tools(self, tools: List[ToolSchema]) -> List[Dict[str, Any]]:
        """Format tools for OpenAI API."""
        return [tool.to_openai_format() for tool in tools]
    
    def parse_tool_calls(self, response: Dict[str, Any]) -> List[ToolCall]:
        """Parse tool calls from OpenAI response (both streaming and non-streaming)."""
        tool_calls = []
        
        if "choices" in response and response["choices"]:
            choice = response["choices"][0]
            
            # Check for tool calls in message (non-streaming)
            message = choice.get("message", {})
            if "tool_calls" in message:
                for tc in message["tool_calls"]:
                    if tc.get("function", {}).get("arguments"):
                        try:
                            tool_calls.append(ToolCall(
                                id=tc["id"],
                                tool_name=tc["function"]["name"],
                                arguments=json.loads(tc["function"]["arguments"])
                            ))
                        except (json.JSONDecodeError, KeyError) as e:
                            print(f"🔧 DEBUG: Failed to parse tool call arguments: {e}")
            
            # Check for tool calls in delta (streaming)
            delta = choice.get("delta", {})
            if "tool_calls" in delta:
                for tc in delta["tool_calls"]:
                    # In streaming, we need to accumulate arguments across chunks
                    # For now, only process complete tool calls
                    if (tc.get("function", {}).get("name") and
                        tc.get("function", {}).get("arguments") and
                        tc.get("id")):
                        try:
                            # Try to parse arguments - they might be partial in streaming
                            args_str = tc["function"]["arguments"]
                            if args_str.strip().endswith("}"):  # Complete JSON
                                tool_calls.append(ToolCall(
                                    id=tc["id"],
                                    tool_name=tc["function"]["name"],
                                    arguments=json.loads(args_str)
                                ))
                        except (json.JSONDecodeError, KeyError) as e:
                            print(f"🔧 DEBUG: Failed to parse streaming tool call arguments: {e}")
        
        return tool_calls


class AnthropicProvider(BaseLLMProvider):
    """Anthropic API provider."""
    
    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(api_key, base_url, **kwargs)
        self.client = None
    
    async def _get_client(self):
        """Get or create Anthropic client."""
        if self.client is None:
            try:
                import anthropic
                
                self.client = anthropic.AsyncAnthropic(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
            except ImportError:
                error = LLMError(
                    error_type=ErrorType.SYSTEM_ERROR,
                    message="Anthropic package not installed. Install with: pip install anthropic",
                    recoverable=False
                )
                raise LLMException(error)
                raise LLMException(error)
        
        return self.client
    
    async def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Complete a non-streaming Anthropic request."""
        client = await self._get_client()
        
        try:
            # Convert messages format for Anthropic
            system_message = None
            anthropic_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    anthropic_messages.append(msg)
            
            params = {
                "model": kwargs.get("model", "claude-3-sonnet-20240229"),
                "messages": anthropic_messages,
                "max_tokens": kwargs.get("max_tokens", 4000),
                "temperature": kwargs.get("temperature", 0.7),
            }
            
            if system_message:
                params["system"] = system_message
            
            if tools:
                params["tools"] = tools
            
            response = await client.messages.create(**params)
            return response.model_dump()
            
        except Exception as e:
            error = LLMError(
                error_type=ErrorType.API_ERROR,
                message=f"Anthropic API error: {str(e)}",
                details={"provider": "anthropic", "model": kwargs.get("model", "claude-3-sonnet-20240229")},
                recoverable=True
            )
            raise LLMException(error)
    
    async def stream_complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """Complete a streaming Anthropic request."""
        client = await self._get_client()
        
        try:
            # Convert messages format for Anthropic
            system_message = None
            anthropic_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    anthropic_messages.append(msg)
            
            params = {
                "model": kwargs.get("model", "claude-3-sonnet-20240229"),
                "messages": anthropic_messages,
                "max_tokens": kwargs.get("max_tokens", 4000),
                "temperature": kwargs.get("temperature", 0.7),
                "stream": True,
            }
            
            if system_message:
                params["system"] = system_message
            
            if tools:
                params["tools"] = tools
            
            async for chunk in await client.messages.create(**params):
                yield chunk.model_dump()
                
        except Exception as e:
            error = LLMError(
                error_type=ErrorType.API_ERROR,
                message=f"Anthropic streaming error: {str(e)}",
                details={"provider": "anthropic", "model": kwargs.get("model", "claude-3-sonnet-20240229")},
                recoverable=True
            )
            raise LLMException(error)
    
    def format_tools(self, tools: List[ToolSchema]) -> List[Dict[str, Any]]:
        """Format tools for Anthropic API."""
        return [tool.to_anthropic_format() for tool in tools]
    
    def parse_tool_calls(self, response: Dict[str, Any]) -> List[ToolCall]:
        """Parse tool calls from Anthropic response."""
        tool_calls = []
        
        if "content" in response:
            for content_block in response["content"]:
                if content_block.get("type") == "tool_use":
                    tool_calls.append(ToolCall(
                        id=content_block["id"],
                        tool_name=content_block["name"],
                        arguments=content_block["input"]
                    ))
        
        return tool_calls


class LLMClient:
    """
    Generic LLM client that supports multiple providers with streaming.
    """
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        self.provider_name = provider
        self.model = model
        self.config = kwargs
        
        # Initialize provider
        if provider == "openai":
            self.provider = OpenAIProvider(api_key, base_url, **kwargs)
        elif provider == "anthropic":
            self.provider = AnthropicProvider(api_key, base_url, **kwargs)
        else:
            error = LLMError(
                error_type=ErrorType.SYSTEM_ERROR,
                message=f"Unsupported provider: {provider}",
                recoverable=False
            )
    
    async def complete(
        self,
        request: GenericRequest,
        tools: Optional[List[ToolSchema]] = None
    ) -> GenericResponse:
        """Complete a non-streaming request."""
        
        # Build messages - use conversation history if available, otherwise build from prompts
        if request.messages:
            # Use existing conversation history (includes tool results)
            # Clean messages to ensure JSON serialization compatibility
            messages = json.loads(json.dumps(request.messages, default=json_serializer))
        else:
            # Build initial conversation from prompts
            messages = [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_message}
            ]
        
        # Format tools if provided
        formatted_tools = None
        if tools and request.tools_enabled:
            formatted_tools = self.provider.format_tools(tools)
        
        # Build parameters
        params = {
            "model": request.model or self.model,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "frequency_penalty": request.frequency_penalty,
            "presence_penalty": request.presence_penalty,
            "tool_choice": request.tool_choice,
        }
        
        try:
            # Make API call
            response = await self.provider.complete(
                messages=messages,
                tools=formatted_tools,
                **params
            )
            
            # Debug: Print raw LLM response
            print(f"🔧 DEBUG: Raw LLM Response:")
            print(json.dumps(response, indent=2)[:1000] + "..." if len(json.dumps(response)) > 1000 else json.dumps(response, indent=2))
            
            # Parse response
            content = self._extract_content(response)
            tool_calls = self.provider.parse_tool_calls(response)
            
            # Debug: Print parsed content and tool calls
            print(f"🔧 DEBUG: Parsed content: {repr(content)}")
            print(f"🔧 DEBUG: Parsed tool calls: {len(tool_calls)} calls")
            for tc in tool_calls:
                print(f"  - {tc.tool_name}({tc.arguments})")
            
            # Create metadata
            metadata = ResponseMetadata(
                request_id=request.id,
                session_id="",  # Will be set by session
                model_used=request.model or self.model,
                provider=self.provider_name,
                prompt_tokens=response.get("usage", {}).get("prompt_tokens"),
                completion_tokens=response.get("usage", {}).get("completion_tokens"),
                total_tokens=response.get("usage", {}).get("total_tokens"),
            )
            metadata.mark_complete()
            
            return GenericResponse.success_response(
                request_id=request.id,
                result={"content": content, "tool_calls": [tc.dict() for tc in tool_calls]},
                raw_content=content,
                metadata=metadata
            )
            
        except LLMException:
            raise
        except Exception as e:
            error = LLMError(
                error_type=ErrorType.UNKNOWN_ERROR,
                message=f"Unexpected error in LLM completion: {str(e)}",
                details={"provider": self.provider_name, "model": self.model},
                recoverable=True
            )
            raise LLMException(error)
    
    async def stream_complete(
        self,
        request: GenericRequest,
        tools: Optional[List[ToolSchema]] = None
    ) -> AsyncIterator[StreamEvent]:
        """Complete a streaming request."""
        
        # Build messages - use conversation history if available, otherwise build from prompts
        if request.messages:
            # Use existing conversation history (includes tool results)
            # Clean messages to ensure JSON serialization compatibility
            messages = json.loads(json.dumps(request.messages, default=json_serializer))
        else:
            # Build initial conversation from prompts
            messages = [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_message}
            ]
        
        # Format tools if provided
        formatted_tools = None
        if tools and request.tools_enabled:
            formatted_tools = self.provider.format_tools(tools)
        
        # Build parameters
        params = {
            "model": request.model or self.model,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "tool_choice": request.tool_choice,
        }
        
        try:
            accumulated_content = ""
            # Track tool calls being built across chunks
            tool_call_builders = {}
            
            async for chunk in self.provider.stream_complete(
                messages=messages,
                tools=formatted_tools,
                **params
            ):
                # No debug output for regular content chunks - handled by test script
                
                # Parse chunk content
                delta_content = self._extract_delta_content(chunk)
                
                if delta_content:
                    accumulated_content += delta_content
                    
                    yield StreamEvent.content_chunk(
                        request_id=request.id,
                        session_id="",  # Will be set by session
                        content=accumulated_content,
                        delta=delta_content
                    )
                
                # Handle streaming tool calls
                if "choices" in chunk and chunk["choices"]:
                    choice = chunk["choices"][0]
                    delta = choice.get("delta", {})
                    
                    if "tool_calls" in delta and delta["tool_calls"] is not None:
                        for tc_delta in delta["tool_calls"]:
                            index = tc_delta.get("index", 0)
                            
                            # Initialize tool call builder if needed
                            if index not in tool_call_builders:
                                tool_call_builders[index] = {
                                    "id": None,
                                    "name": None,
                                    "arguments": ""
                                }
                            
                            builder = tool_call_builders[index]
                            
                            # Update tool call data
                            if tc_delta.get("id"):
                                builder["id"] = tc_delta["id"]
                            
                            if tc_delta.get("function"):
                                func = tc_delta["function"]
                                if func.get("name"):
                                    builder["name"] = func["name"]
                                if func.get("arguments"):
                                    builder["arguments"] += func["arguments"]
                    
                    # Check if we have complete tool calls
                    finish_reason = choice.get("finish_reason")
                    if finish_reason == "tool_calls":
                        for index, builder in tool_call_builders.items():
                            if builder["id"] and builder["name"] and builder["arguments"]:
                                try:
                                    arguments = json.loads(builder["arguments"])
                                    tool_call = ToolCall(
                                        id=builder["id"],
                                        tool_name=builder["name"],
                                        arguments=arguments
                                    )
                                    yield StreamEvent(
                                        event_type=StreamEventType.TOOL_CALL_START,
                                        request_id=request.id,
                                        session_id="",
                                        tool_call=tool_call
                                    )
                                except json.JSONDecodeError as e:
                                    print(f"❌ Failed to parse tool call arguments: {e}")
            
        except LLMException:
            raise
        except Exception as e:
            error = LLMError(
                error_type=ErrorType.UNKNOWN_ERROR,
                message=f"Unexpected error in LLM streaming: {str(e)}",
                details={"provider": self.provider_name, "model": self.model},
                recoverable=True
            )
            raise LLMException(error)
    
    def _extract_content(self, response: Dict[str, Any]) -> str:
        """Extract content from provider response."""
        if self.provider_name == "openai":
            if "choices" in response and response["choices"]:
                return response["choices"][0].get("message", {}).get("content", "")
        elif self.provider_name == "anthropic":
            if "content" in response:
                for content_block in response["content"]:
                    if content_block.get("type") == "text":
                        return content_block.get("text", "")
        
        return ""
    
    def _extract_delta_content(self, chunk: Dict[str, Any]) -> str:
        """Extract delta content from streaming chunk."""
        if self.provider_name == "openai":
            if "choices" in chunk and chunk["choices"]:
                delta = chunk["choices"][0].get("delta", {})
                return delta.get("content", "")
        elif self.provider_name == "anthropic":
            if chunk.get("type") == "content_block_delta":
                return chunk.get("delta", {}).get("text", "")
        
        return ""
    
    async def close(self) -> None:
        """Close the client and clean up resources."""
        # Close provider-specific resources if needed
        if hasattr(self.provider, 'client') and self.provider.client:
            if hasattr(self.provider.client, 'close'):
                await self.provider.client.close()