#!/usr/bin/env python3
"""
MCP Server runner script.

This script runs the FastMCP server with proper path configuration.
"""

import argparse
import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the server
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from llm_mcp.mcp.server.server import FastMCPServer
from llm_mcp.models.semantic import SemanticSearchConfig, QdrantConfig, EmbeddingModel


def main():
    """Run the FastMCP server."""
    parser = argparse.ArgumentParser(description="Run the FastMCP server")
    parser.add_argument(
        "--qdrant-location",
        type=str,
        help="Qdrant database location for semantic search"
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="code_chunks",
        help="Qdrant collection name"
    )
    parser.add_argument(
        "--vector-size",
        type=int,
        default=384,
        help="Vector dimension size"
    )
    args = parser.parse_args()
    
    # Check for qdrant location from CLI args or environment variable
    qdrant_location = args.qdrant_location or os.getenv("QDRANT_LOCATION")
    
    semantic_config = None
    if qdrant_location and qdrant_location != ":memory:":
        # Create semantic search configuration with provided arguments
        qdrant_config = QdrantConfig(
            location=qdrant_location,
            collection_name=args.collection_name,
            vector_size=args.vector_size
        )
        embedding_model = EmbeddingModel.minilm()  # Use lightweight model by default
        
        semantic_config = SemanticSearchConfig(
            qdrant_config=qdrant_config,
            embedding_model=embedding_model
        )
        print(f"Semantic search enabled with Qdrant location: {qdrant_location}")
        print(f"Collection: {args.collection_name}, Vector size: {args.vector_size}")
    else:
        print("Semantic search disabled. Provide --qdrant-location or set QDRANT_LOCATION environment variable to enable.")
    
    server = FastMCPServer(semantic_config=semantic_config)
    server.run()


if __name__ == "__main__":
    main()