#!/usr/bin/env python3
"""
Enhanced version of test_llm_resp.py that demonstrates ephemeral semantic database integration.

This example shows how to:
1. Create an ephemeral semantic database from the current repository
2. Start an MCP server with semantic search enabled
3. Run LLM analysis with semantic search capabilities
4. Clean up the ephemeral database when done
"""

import asyncio
import os
import sys
from pathlib import Path
from pydantic import BaseModel, Field

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_mcp import LLMSession, GenericRequest
from llm_mcp.models import LLMResp
from llm_mcp.rag import EphemeralSemanticDB, create_ephemeral_semantic_db
from llm_mcp.mcp.server.server import FastMCPServer
from llm_mcp.mcp.server.process import MCPServerProcess


# Example Pydantic model for structured response
class CodeAnalysis(BaseModel):
    """Example model for code analysis results."""
    summary: str = Field(..., description="Brief summary of the code")
    files_analyzed: list[str] = Field(..., description="List of files that were analyzed")
    key_findings: list[str] = Field(..., description="Key findings from the analysis")
    recommendations: list[str] = Field(..., description="Recommendations for improvement")
    semantic_insights: list[str] = Field(default_factory=list, description="Insights from semantic search")


async def test_llm_resp_with_semantic():
    """Test LLM response with ephemeral semantic database."""
    print("🚀 Testing LLMResp with Ephemeral Semantic Database")
    print("=" * 60)
    
    # Step 1: Create ephemeral semantic database
    print("📊 Step 1: Creating ephemeral semantic database...")
    async with create_ephemeral_semantic_db(
        repository_path=".",
        embedding_model="minilm",  # Use lightweight model for demo
        languages=["python"]  # Focus on Python files
    ) as ephemeral_db:
        
        print(f"✅ Created ephemeral database at: {ephemeral_db.database_path}")
        print(f"📈 Stats: {ephemeral_db.stats}")
        
        # Step 2: Get semantic configuration
        semantic_config = ephemeral_db.get_semantic_config()
        if not semantic_config:
            print("❌ Failed to get semantic configuration")
            return
        
        print(f"🔧 Semantic config: {semantic_config.qdrant_config.location}")
        
        # Step 3: Start MCP server with semantic search
        print("\n📊 Step 2: Starting MCP server with semantic search...")
        server_process = None
        try:
            server_process = MCPServerProcess(
                server_name="semantic-analysis-server",
                semantic_config=semantic_config
            )
            await server_process.start()
            print("✅ MCP server started with semantic search enabled")
            
            # Step 4: Initialize LLM session
            print("\n📊 Step 3: Initializing LLM session...")
            session = LLMSession(
                provider="openai",
                model="anthropic/claude-3.7-sonnet",
                api_key="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",
                base_url="https://openrouter.ai/api/v1"
            )
            
            # Step 5: Create enhanced request with semantic search
            request_with_semantic = GenericRequest(
                system_prompt="""You are a code analyst with access to both file analysis tools AND semantic search capabilities. You MUST use these tools to analyze files.

CRITICAL: You cannot analyze files without using tools. You have these tools available:
- list_directory: to see what files exist
- read_file: to read file contents
- search_in_files: to search for patterns
- working_semantic_search: to perform semantic search across the codebase

MANDATORY WORKFLOW:
1. IMMEDIATELY use list_directory tool with directory_path="." to see files
2. Use working_semantic_search tool to find relevant code patterns semantically
3. Use read_file tool to examine specific files found through semantic search
4. Use search_in_files tool for additional pattern matching if needed
5. Provide analysis in JSON format:

{
    "summary": "Brief summary of the code",
    "files_analyzed": ["list", "of", "files"],
    "key_findings": ["finding1", "finding2"],
    "recommendations": ["rec1", "rec2"],
    "semantic_insights": ["insight1", "insight2"]
}

IMPORTANT: Start with list_directory, then use working_semantic_search to find authentication, security, or injection-related code patterns.""",
                
                user_message="""Please analyze the Python files in this directory with focus on security vulnerabilities. 

You MUST start by using the list_directory tool, then use working_semantic_search to find:
1. Authentication and authorization code
2. Input validation patterns
3. Database query construction
4. Command execution patterns
5. File handling operations

Use semantic search queries like:
- "authentication login security"
- "sql injection database query"
- "command injection subprocess"
- "file upload validation"

Then examine the found files in detail.""",
                context={"task": "semantic_security_analysis"},
                stream=True,
                tool_choice="auto"
            )
            
            print("\n📊 Step 4: Running semantic analysis...")
            async with session:
                print("✅ Session initialized!")
                
                # Get structured response with semantic analysis
                result: LLMResp[CodeAnalysis] = await session.analyze_with_response(
                    request_with_semantic,
                    response_model=CodeAnalysis,
                    display=True  # Show streaming output
                )
                
                print("\n" + "=" * 60)
                print("📋 SEMANTIC ANALYSIS RESULTS")
                print("=" * 60)
                
                print(f"\n🔍 Raw Response Length: {len(result.raw_response)} characters")
                print(f"📝 Clean Text Length: {len(result.clean_text)} characters")
                print(f"✅ Parsing Success: {result.parsing_success}")
                
                if result.parsing_error:
                    print(f"❌ Parsing Error: {result.parsing_error}")
                
                if result.json_extracted:
                    print(f"\n📊 JSON Extracted: {len(result.json_extracted)} fields")
                    for key in result.json_extracted.keys():
                        print(f"   - {key}")
                
                if result.parsed_data:
                    print(f"\n🎯 Parsed Data (CodeAnalysis):")
                    print(f"   Summary: {result.parsed_data.summary[:100]}...")
                    print(f"   Files Analyzed: {result.parsed_data.files_analyzed}")
                    print(f"   Key Findings: {len(result.parsed_data.key_findings)} items")
                    print(f"   Recommendations: {len(result.parsed_data.recommendations)} items")
                    print(f"   Semantic Insights: {len(result.parsed_data.semantic_insights)} items")
                    
                    if result.parsed_data.semantic_insights:
                        print(f"\n🔍 Semantic Insights:")
                        for i, insight in enumerate(result.parsed_data.semantic_insights, 1):
                            print(f"   {i}. {insight}")
                
                print(f"\n📄 Clean Text Preview:")
                print("-" * 30)
                print(result.clean_text[:500] + "..." if len(result.clean_text) > 500 else result.clean_text)
        
        finally:
            # Step 6: Cleanup
            print("\n📊 Step 5: Cleaning up...")
            if server_process:
                await server_process.stop()
                print("✅ MCP server stopped")
            
            # Ephemeral database will be cleaned up automatically when exiting context
            print("✅ Ephemeral database will be cleaned up automatically")


async def test_without_semantic():
    """Test the same analysis without semantic search for comparison."""
    print("\n" + "=" * 60)
    print("🔄 COMPARISON: Running without semantic search")
    print("=" * 60)
    
    session = LLMSession(
        provider="openai",
        model="anthropic/claude-3.7-sonnet",
        api_key="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",
        base_url="https://openrouter.ai/api/v1"
    )
    
    request_without_semantic = GenericRequest(
        system_prompt="""You are a code analyst with access to file analysis tools (but NO semantic search). You have these tools:
- list_directory: to see what files exist
- read_file: to read file contents  
- search_in_files: to search for patterns

Analyze the code and provide results in JSON format.""",
        user_message="Please analyze the Python files for security vulnerabilities using only traditional file operations.",
        context={"task": "traditional_analysis"},
        stream=True,
        tool_choice="auto"
    )
    
    async with session:
        result = await session.analyze_with_response(
            request_without_semantic,
            response_model=CodeAnalysis,
            display=True
        )
        
        print(f"\n📊 Traditional Analysis Results:")
        print(f"   Files Found: {len(result.parsed_data.files_analyzed) if result.parsed_data else 0}")
        print(f"   Findings: {len(result.parsed_data.key_findings) if result.parsed_data else 0}")


if __name__ == "__main__":
    print("🎯 Semantic Search Enhanced LLM Analysis Demo")
    print("This demo shows the difference between traditional file analysis and semantic search.")
    print()
    
    # Run semantic analysis
    asyncio.run(test_llm_resp_with_semantic())
    
    # Optionally run comparison
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        asyncio.run(test_without_semantic())
    else:
        print("\n💡 Tip: Run with --compare to see traditional analysis comparison")