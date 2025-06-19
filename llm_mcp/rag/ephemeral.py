"""
Utilities for managing ephemeral semantic databases.

This module provides utilities for creating, using, and cleaning up
ephemeral semantic databases that are created in /dev/shm for fast access.
"""

import asyncio
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from .ingestion import SemanticIngester
from ..models.semantic import SemanticSearchConfig, QdrantConfig, EmbeddingModel
from ..utils.constants import DEFAULT_EMBEDDING_MODEL, MAX_FILE_SIZE_BYTES, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

logger = logging.getLogger(__name__)


class EphemeralSemanticDB:
    """
    Context manager for ephemeral semantic databases.
    
    This class handles the lifecycle of ephemeral semantic databases:
    1. Creates the database via ingestion
    2. Provides the database path for use with MCP server
    3. Cleans up the database when done
    """
    
    def __init__(
        self,
        repository_path: str = ".",
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        languages: Optional[list] = None,
        max_file_size: int = MAX_FILE_SIZE_BYTES,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    ):
        """
        Initialize ephemeral database configuration.
        
        Args:
            repository_path: Path to repository to ingest
            embedding_model: Embedding model to use
            languages: Programming languages to include
            max_file_size: Maximum file size to process
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.repository_path = repository_path
        self.embedding_model = embedding_model
        self.languages = languages
        self.max_file_size = max_file_size
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.database_path: Optional[str] = None
        self.ingestion_response: Optional[Any] = None
        self._cleanup_registered = False
    
    async def __aenter__(self):
        """Create the ephemeral database."""
        logger.info(f"Creating ephemeral semantic database for: {self.repository_path}")
        
        # Execute ingestion
        ingester = SemanticIngester()
        
        self.ingestion_response = await ingester.ingest_repository(
            repository_path=self.repository_path,
            database_path=None,  # Let it create ephemeral path
            embedding_model=self.embedding_model,
            languages=self.languages,
            max_file_size=self.max_file_size,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        if not self.ingestion_response.success:
            raise RuntimeError(f"Failed to create ephemeral database: {self.ingestion_response.error_message}")
        
        self.database_path = self.ingestion_response.database_path
        logger.info(f"Created ephemeral database at: {self.database_path}")
        
        # Register cleanup on exit
        if not self._cleanup_registered:
            import atexit
            atexit.register(self._cleanup_sync)
            self._cleanup_registered = True
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up the ephemeral database."""
        await self.cleanup()
    
    async def cleanup(self):
        """Clean up the ephemeral database."""
        if self.database_path and Path(self.database_path).exists():
            try:
                shutil.rmtree(self.database_path)
                logger.info(f"Cleaned up ephemeral database: {self.database_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup ephemeral database {self.database_path}: {e}")
            finally:
                self.database_path = None
    
    def _cleanup_sync(self):
        """Synchronous cleanup for atexit."""
        if self.database_path and Path(self.database_path).exists():
            try:
                shutil.rmtree(self.database_path)
                logger.info(f"Cleaned up ephemeral database on exit: {self.database_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup ephemeral database on exit: {e}")
    
    def get_semantic_config(self) -> Optional[SemanticSearchConfig]:
        """
        Get semantic search configuration for the ephemeral database.
        
        Returns:
            SemanticSearchConfig that can be used with MCP server
        """
        if not self.database_path or not self.ingestion_response:
            return None
        
        # Get embedding model configuration
        from ..utils.constants import EMBEDDING_MODELS
        
        if self.embedding_model in EMBEDDING_MODELS:
            model_config = EMBEDDING_MODELS[self.embedding_model]
            embedding_model = EmbeddingModel(
                name=self.embedding_model,
                model_path=model_config["model_path"],
                dimension=model_config["dimension"],
                max_length=model_config["max_length"]
            )
        else:
            # Fallback to default
            model_config = EMBEDDING_MODELS[DEFAULT_EMBEDDING_MODEL]
            embedding_model = EmbeddingModel(
                name=DEFAULT_EMBEDDING_MODEL,
                model_path=model_config["model_path"],
                dimension=model_config["dimension"],
                max_length=model_config["max_length"]
            )
        
        # Create Qdrant configuration
        qdrant_config = QdrantConfig(
            location=self.database_path,
            collection_name=self.ingestion_response.collection_name,
            vector_size=self.ingestion_response.vector_dimension
        )
        
        # Create semantic search configuration
        return SemanticSearchConfig(
            embedding_model=embedding_model,
            qdrant_config=qdrant_config,
            max_file_size=self.max_file_size,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get ingestion statistics."""
        if not self.ingestion_response:
            return {}
        
        return {
            "database_path": self.database_path,
            "collection_name": self.ingestion_response.collection_name,
            "embedding_model": self.ingestion_response.embedding_model_used,
            "vector_dimension": self.ingestion_response.vector_dimension,
            "files_processed": self.ingestion_response.indexing_stats.files_processed,
            "chunks_created": self.ingestion_response.indexing_stats.chunks_created,
            "database_size": self.ingestion_response.total_database_size,
            "ingestion_time": self.ingestion_response.ingestion_time
        }


@asynccontextmanager
async def create_ephemeral_semantic_db(
    repository_path: str = ".",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    languages: Optional[list] = None
):
    """
    Async context manager for creating ephemeral semantic databases.
    
    Usage:
        async with create_ephemeral_semantic_db(".") as db:
            config = db.get_semantic_config()
            # Use config with MCP server
            server = FastMCPServer(semantic_config=config)
            # Database is automatically cleaned up when exiting context
    """
    db = EphemeralSemanticDB(
        repository_path=repository_path,
        embedding_model=embedding_model,
        languages=languages
    )
    
    async with db:
        yield db


def cleanup_all_ephemeral_dbs():
    """Clean up all ephemeral databases in /dev/shm and temp directories."""
    cleanup_paths = []
    
    # Check /dev/shm
    shm_path = Path("/dev/shm/semantic_databases")
    if shm_path.exists():
        cleanup_paths.append(shm_path)
    
    # Check temp directory
    temp_path = Path(tempfile.gettempdir()) / "semantic_databases"
    if temp_path.exists():
        cleanup_paths.append(temp_path)
    
    for cleanup_path in cleanup_paths:
        try:
            for db_dir in cleanup_path.iterdir():
                if db_dir.is_dir() and "ephemeral_semantic" in db_dir.name:
                    try:
                        shutil.rmtree(db_dir)
                        logger.info(f"Cleaned up ephemeral database: {db_dir}")
                    except Exception as e:
                        logger.warning(f"Failed to cleanup {db_dir}: {e}")
        except Exception as e:
            logger.warning(f"Failed to cleanup ephemeral databases in {cleanup_path}: {e}")