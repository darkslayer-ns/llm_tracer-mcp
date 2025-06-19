"""
RAG (Retrieval-Augmented Generation) module for semantic search and ingestion.

This module provides functionality for:
- Creating ephemeral semantic databases
- Ingesting repositories into vector databases
- Managing database lifecycle
"""

from .ingestion import SemanticIngester
from .ephemeral import EphemeralSemanticDB, create_ephemeral_semantic_db

__all__ = [
    "SemanticIngester",
    "EphemeralSemanticDB", 
    "create_ephemeral_semantic_db"
]