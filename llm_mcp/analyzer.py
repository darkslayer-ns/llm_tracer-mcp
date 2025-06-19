"""
LLM Repository Analyzer

A simple function that analyzes repositories using LLM with semantic search capabilities.
"""

import json
import os
from typing import Type, TypeVar, Optional, Union, Tuple, Dict, Any
from pydantic import BaseModel

from .llm.session import LLMSession
from .models.requests import GenericRequest
from .models.llm_response import LLMResp
from .rag import create_ephemeral_semantic_db
from .rag.ephemeral import EphemeralSemanticDB

T = TypeVar('T', bound=BaseModel)

# Global singleton for MCP manager to prevent database conflicts
_global_mcp_manager: Optional[Any] = None
_global_semantic_config: Optional[Any] = None


async def cleanup_global_mcp_manager():
    """Clean up the global MCP manager when semantic database is being destroyed."""
    global _global_mcp_manager, _global_semantic_config
    
    if _global_mcp_manager is not None:
        print("🔄 Cleaning up global MCP manager...")
        try:
            await _global_mcp_manager.close()
            print("✅ Global MCP manager cleaned up")
        except Exception as e:
            print(f"⚠️  Warning: Error cleaning up MCP manager: {e}")
        finally:
            _global_mcp_manager = None
            _global_semantic_config = None


def _get_model_json_schema(response_model: Type[BaseModel]) -> str:
    """Get JSON schema from Pydantic model."""
    return json.dumps(response_model.model_json_schema(), indent=2)


async def analyze_repository_with_llm(
    system_prompt: str,
    user_message: str,
    response_model: Type[T],
    repository_path: str = ".",
    display: bool = True,
    semantic_db: Optional[EphemeralSemanticDB] = None,
    return_db: bool = False
) -> Union[T, Tuple[T, EphemeralSemanticDB]]:
    """
    Analyze a repository using LLM with semantic search capabilities.
    
    Args:
        system_prompt: The system prompt that defines the LLM's behavior and instructions
        user_message: The user message/query for the analysis
        response_model: Pydantic model class for structured response parsing
        repository_path: Path to the repository to analyze (default: current directory)
        display: Whether to show streaming output (default: True)
        semantic_db: Optional existing semantic database to reuse (avoids re-embedding)
        return_db: Whether to return the semantic database for reuse (default: False)
    
    Returns:
        If return_db=False: Parsed Pydantic model instance with the analysis results
        If return_db=True: Tuple of (parsed_data, semantic_db) for reuse in subsequent calls
        
    Raises:
        Exception: If analysis fails or parsing is unsuccessful
    """
    print("🚀 Starting LLM Analysis with Semantic Search")
    print("=" * 60)
    
    # Use existing semantic database or create new one
    if semantic_db is not None:
        print("📊 Step 1: Reusing existing semantic database...")
        print(f"✅ Using database: {semantic_db.database_path}")
        print(f"📈 Cached stats: {semantic_db.stats}")
        
        # Get semantic configuration for MCP server
        semantic_config = semantic_db.get_semantic_config()
        
        # Perform analysis without cleanup
        result = await _perform_analysis(
            system_prompt, user_message, response_model, semantic_config, display
        )
        
        if return_db:
            return result, semantic_db
        else:
            return result
    
    else:
        # Create new ephemeral semantic database
        print("📊 Step 1: Creating ephemeral semantic database...")
        
        if return_db:
            # Don't use context manager - caller will handle cleanup
            semantic_db = EphemeralSemanticDB(
                repository_path=repository_path,
                embedding_model="minilm"
            )
            await semantic_db.__aenter__()
            
            print(f"✅ Created ephemeral database: {semantic_db.database_path}")
            print(f"📈 Ingestion stats: {semantic_db.stats}")
            
            # Get semantic configuration for MCP server
            semantic_config = semantic_db.get_semantic_config()
            
            # Perform analysis
            result = await _perform_analysis(
                system_prompt, user_message, response_model, semantic_config, display
            )
            
            return result, semantic_db
        
        else:
            # Use context manager for automatic cleanup
            async with create_ephemeral_semantic_db(
                repository_path=repository_path,
                embedding_model="minilm"
            ) as semantic_db:
                
                print(f"✅ Created ephemeral database: {semantic_db.database_path}")
                print(f"📈 Ingestion stats: {semantic_db.stats}")
                
                # Get semantic configuration for MCP server
                semantic_config = semantic_db.get_semantic_config()
                
                # Perform analysis
                result = await _perform_analysis(
                    system_prompt, user_message, response_model, semantic_config, display
                )
                
                print("\n✅ Ephemeral database will be cleaned up automatically")
                
                # Clean up global MCP manager when database is being destroyed
                await cleanup_global_mcp_manager()
                
                return result


async def _get_or_create_mcp_manager(semantic_config) -> Tuple[Any, bool]:
    """
    Get existing MCP manager or create new one if needed.
    
    Returns:
        Tuple of (mcp_manager, is_new_manager)
    """
    global _global_mcp_manager, _global_semantic_config
    
    # Check if we can reuse existing manager
    if (_global_mcp_manager is not None and
        _global_semantic_config is not None and
        _global_semantic_config.database_path == semantic_config.database_path):
        print("♻️  Reusing existing MCP manager for same database")
        return _global_mcp_manager, False
    
    # Clean up old manager if it exists
    if _global_mcp_manager is not None:
        print("🔄 Cleaning up previous MCP manager...")
        try:
            await _global_mcp_manager.close()
        except Exception as e:
            print(f"⚠️  Warning: Error closing previous MCP manager: {e}")
    
    # Create new manager
    print("🆕 Creating new MCP manager...")
    from .mcp.manager import MCPManager
    _global_mcp_manager = MCPManager(semantic_config=semantic_config)
    _global_semantic_config = semantic_config
    
    return _global_mcp_manager, True


async def _perform_analysis(
    system_prompt: str,
    user_message: str,
    response_model: Type[T],
    semantic_config,
    display: bool
) -> T:
    """Internal function to perform the actual LLM analysis."""
    
    print("\n📊 Step 2: Initializing LLM session with semantic search...")
    
    # Get or create singleton MCP manager
    mcp_manager, is_new_manager = await _get_or_create_mcp_manager(semantic_config)
    
    # Get API key from environment variables
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "No API key found. Please set either OPENROUTER_API_KEY or OPENAI_API_KEY environment variable. "
            "For OpenRouter: export OPENROUTER_API_KEY='your-key-here'"
        )
    
    session = LLMSession(
        provider="openai",
        model="anthropic/claude-3.7-sonnet",
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        semantic_config=semantic_config
    )

    # Generate system prompt with JSON schema and automatic tool guidance
    tool_guidance = """You are a application security professional tasked with codebase assesment.

AVAILABLE TOOLS AND USAGE:
- list_directory(directory_path="."): List files and directories
- working_semantic_search(query="search terms", file_patterns=["*.py", "*.js"]): Semantic search (file_patterns must be a list)
- read_file(file_path="path/to/file.py"): Read file contents
- search_in_files(search_term="pattern", file_extensions=[".py", ".js"]): Search with regex (file_extensions must be a list)

CRITICAL: Always pass file_patterns and file_extensions as lists, not strings!
Example: file_patterns=["*.py"] NOT file_patterns="*.py"

Provide final output in JSON only, ommiting any explanations or other human formatable output:

{{ json_schema }}
"""
    
    final_system_prompt = system_prompt.format(json_schema=_get_model_json_schema(response_model)) + tool_guidance

    # Create the request
    request = GenericRequest(
        system_prompt=final_system_prompt,
        user_message=user_message,
        context={"task": "structured_analysis"},
        stream=True,
        tool_choice="auto"
    )

    print("\n🔄 Initializing session...")
    async with session:
        print("✅ Session initialized!")
        print(f"\n📊 Running Analysis with {response_model.__name__} Response Model")
        print("-" * 50)
        
        try:
            # Get structured response with display
            result: LLMResp[T] = await session.analyze_with_response(
                request,
                response_model=response_model,
                display=display
            )
            
            print("\n" + "=" * 60)
            print("📋 ANALYSIS RESULTS")
            print("=" * 60)
            
            print(f"\n🔍 Raw Response Length: {len(result.raw_response)} characters")
            print(f"📝 Clean Text Length: {len(result.clean_text)} characters")
            print(f"✅ Parsing Success: {result.parsing_success}")
            
            if result.parsing_error:
                print(f"❌ Parsing Error: {result.parsing_error}")
                raise Exception(f"Failed to parse response: {result.parsing_error}")
            
            if not result.parsing_success:
                raise Exception("Analysis failed - parsing was not successful")
            
            if not result.parsed_data:
                raise Exception("No parsed data returned from analysis")
            
            if result.json_extracted:
                print(f"\n📊 JSON Extracted: {len(result.json_extracted)} fields")
                for key in result.json_extracted.keys():
                    print(f"   - {key}")
            
            print(f"\n🎯 Successfully parsed {response_model.__name__} data")
            
            return result.parsed_data
            
        finally:
            # Note: We don't stop the singleton MCP manager here
            # It will be reused for subsequent calls with the same database
            print("\n♻️  Keeping MCP manager alive for potential reuse")