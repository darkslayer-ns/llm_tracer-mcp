"""
Example demonstrating the SemanticSearchTool for code analysis and discovery.
"""

import asyncio
from pathlib import Path

from llm_mcp.tools import SemanticSearchTool
from llm_mcp.models.semantic import (
    SemanticSearchRequest, SemanticSearchConfig, 
    EmbeddingModel, QdrantConfig, ChunkType, SearchMode
)
from llm_mcp.models.tools import ToolExecutionContext


async def main():
    """Demonstrate semantic search capabilities."""
    
    # Configure the semantic search tool
    config = SemanticSearchConfig(
        embedding_model=EmbeddingModel.minilm(),  # Lightweight model for demo
        qdrant_config=QdrantConfig(
            location=":memory:",  # Ephemeral in-memory database
            collection_name="demo_code_search"
        )
    )
    
    # Initialize the tool
    search_tool = SemanticSearchTool(config)
    await search_tool.initialize()
    
    print("🔍 Semantic Search Tool Demo")
    print("=" * 50)
    
    # Example 1: Search for authentication-related code
    print("\n1. Searching for authentication-related code...")
    
    auth_request = SemanticSearchRequest(
        query="user authentication login password validation",
        directory=".",  # Search current directory
        languages=["python", "javascript", "typescript"],
        max_results=5,
        similarity_threshold=0.6,
        include_context=True,
        context_lines=3,
        chunk_types=[ChunkType.FUNCTION, ChunkType.CLASS]
    )
    
    context = ToolExecutionContext(
        session_id="demo_session",
        request_id="auth_search_001"
    )
    
    auth_response = await search_tool.execute(auth_request, context)
    
    if auth_response.success:
        print(f"✅ Found {auth_response.total_results} authentication-related code chunks")
        if auth_response.indexing_stats:
            stats = auth_response.indexing_stats
            print(f"📊 Indexed {stats.files_indexed} files, {stats.chunks_created} chunks")
            print(f"🌐 Languages found: {', '.join(stats.languages_found)}")
        
        for i, chunk in enumerate(auth_response.results[:3], 1):
            print(f"\n📄 Result {i}: {chunk.file_path}:{chunk.start_line}-{chunk.end_line}")
            print(f"🏷️  Type: {chunk.chunk_type.value}, Language: {chunk.language}")
            print(f"⭐ Similarity: {chunk.similarity_score:.3f}")
            if chunk.function_name:
                print(f"🔧 Function: {chunk.function_name}")
            if chunk.class_name:
                print(f"🏗️  Class: {chunk.class_name}")
            print(f"📝 Content preview: {chunk.content[:200]}...")
    else:
        print(f"❌ Search failed: {auth_response.error_message}")
    
    # Example 2: Search for API endpoint handlers
    print("\n\n2. Searching for API endpoint handlers...")
    
    api_request = SemanticSearchRequest(
        query="REST API endpoint route handler HTTP GET POST",
        directory=".",
        languages=["python", "javascript", "go"],
        max_results=3,
        similarity_threshold=0.7,
        search_mode=SearchMode.SEMANTIC,
        chunk_types=[ChunkType.FUNCTION]
    )
    
    api_response = await search_tool.execute(api_request, context)
    
    if api_response.success:
        print(f"✅ Found {api_response.total_results} API-related code chunks")
        
        for i, chunk in enumerate(api_response.results, 1):
            print(f"\n📄 Result {i}: {chunk.file_path}")
            print(f"🔧 Function: {chunk.function_name or 'Unknown'}")
            print(f"⭐ Similarity: {chunk.similarity_score:.3f}")
            print(f"📝 Content: {chunk.content[:150]}...")
    else:
        print(f"❌ API search failed: {api_response.error_message}")
    
    # Example 3: Search for database-related code
    print("\n\n3. Searching for database operations...")
    
    db_request = SemanticSearchRequest(
        query="database connection query SQL insert update delete",
        directory=".",
        max_results=5,
        similarity_threshold=0.65,
        chunk_types=[ChunkType.FUNCTION, ChunkType.CLASS, ChunkType.METHOD]
    )
    
    db_response = await search_tool.execute(db_request, context)
    
    if db_response.success:
        print(f"✅ Found {db_response.total_results} database-related code chunks")
        
        # Group results by file
        files_found = {}
        for chunk in db_response.results:
            if chunk.file_path not in files_found:
                files_found[chunk.file_path] = []
            files_found[chunk.file_path].append(chunk)
        
        print(f"📁 Found relevant code in {len(files_found)} files:")
        for file_path, chunks in files_found.items():
            print(f"  • {file_path}: {len(chunks)} chunks")
    else:
        print(f"❌ Database search failed: {db_response.error_message}")
    
    # Clean up
    await search_tool.close()
    print("\n🎉 Demo completed!")


async def demo_with_custom_directory():
    """Demo searching in a specific directory with custom configuration."""
    
    print("\n" + "=" * 60)
    print("🎯 Custom Directory Search Demo")
    print("=" * 60)
    
    # Custom configuration for larger codebases
    config = SemanticSearchConfig(
        embedding_model=EmbeddingModel.codebert(),  # Better for code understanding
        qdrant_config=QdrantConfig(
            location=":memory:",
            collection_name="custom_search",
            vector_size=768  # CodeBERT dimension
        ),
        chunk_size=1500,  # Larger chunks for more context
        chunk_overlap=300,
        max_workers=2  # Parallel processing
    )
    
    search_tool = SemanticSearchTool(config)
    await search_tool.initialize()
    
    # Search for error handling patterns
    error_request = SemanticSearchRequest(
        query="error handling exception try catch finally logging",
        directory="../",  # Search parent directory
        languages=["python"],
        max_results=8,
        similarity_threshold=0.6,
        include_context=True,
        context_lines=5,
        reindex=True  # Force fresh indexing
    )
    
    context = ToolExecutionContext(
        session_id="custom_demo",
        request_id="error_handling_search"
    )
    
    response = await search_tool.execute(error_request, context)
    
    if response.success:
        print(f"✅ Found {response.total_results} error handling patterns")
        
        if response.indexing_stats:
            stats = response.indexing_stats
            print(f"⏱️  Indexing took {stats.indexing_time:.2f} seconds")
            print(f"📊 Processed {stats.files_processed} files")
        
        print(f"🔍 Search took {response.search_time:.2f} seconds")
        
        # Show detailed results
        for i, chunk in enumerate(response.results[:3], 1):
            print(f"\n📄 Result {i}:")
            print(f"   File: {chunk.file_path}")
            print(f"   Lines: {chunk.start_line}-{chunk.end_line}")
            print(f"   Type: {chunk.chunk_type.value}")
            print(f"   Score: {chunk.similarity_score:.3f}")
            
            if chunk.context_before:
                print("   Context before:")
                for line in chunk.context_before[-2:]:
                    print(f"     {line}")
            
            print("   Content:")
            for line in chunk.content.split('\n')[:5]:
                print(f"     {line}")
            
            if chunk.context_after:
                print("   Context after:")
                for line in chunk.context_after[:2]:
                    print(f"     {line}")
    else:
        print(f"❌ Custom search failed: {response.error_message}")
    
    await search_tool.close()


if __name__ == "__main__":
    print("🚀 Starting Semantic Search Demo")
    print("This demo shows how to use the SemanticSearchTool for code analysis")
    print("Make sure you have the semantic dependencies installed:")
    print("  pip install -e .[semantic]")
    print()
    
    try:
        # Run basic demo
        asyncio.run(main())
        
        # Run custom directory demo
        asyncio.run(demo_with_custom_directory())
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Install semantic search dependencies with:")
        print("  pip install -e .[semantic]")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()