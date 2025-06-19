#!/usr/bin/env python3
"""
Example demonstrating how to run the MCP server with semantic search enabled.

This example shows how to configure and run the MCP server with the WorkingSemanticSearchTool
when a valid qdrant_location is provided.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_mcp.mcp.server.server import create_server, run_server
from llm_mcp.models.semantic import SemanticSearchConfig, QdrantConfig, EmbeddingModel


def run_server_with_semantic_search():
    """Run MCP server with semantic search enabled."""
    
    # Configure Qdrant location (use a persistent directory)
    qdrant_location = "./qdrant_data"
    
    # Create semantic search configuration
    qdrant_config = QdrantConfig(location=qdrant_location)
    embedding_model = EmbeddingModel.minilm()  # Lightweight model
    
    semantic_config = SemanticSearchConfig(
        qdrant_config=qdrant_config,
        embedding_model=embedding_model
    )
    
    print(f"Starting MCP server with semantic search enabled")
    print(f"Qdrant location: {qdrant_location}")
    print(f"Embedding model: {embedding_model.name}")
    print("Available tools will include: working_semantic_search")
    print("\nTo test the semantic search tool, use an MCP client to call:")
    print("working_semantic_search(query='find authentication code', directory='.')")
    
    # Run the server
    run_server(
        server_name="llm-mcp-with-semantic-search",
        semantic_config=semantic_config
    )


def run_server_without_semantic_search():
    """Run MCP server without semantic search (default behavior)."""
    
    print("Starting MCP server without semantic search")
    print("Available tools: search_in_files, read_file, write_file, list_directory")
    
    # Run the server without semantic config
    run_server(server_name="llm-mcp-basic")


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--with-semantic":
        run_server_with_semantic_search()
    else:
        print("Usage:")
        print("  python semantic_search_mcp_example.py                # Run without semantic search")
        print("  python semantic_search_mcp_example.py --with-semantic # Run with semantic search")
        print()
        
        # Default to running without semantic search
        run_server_without_semantic_search()