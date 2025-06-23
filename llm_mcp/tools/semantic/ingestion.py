"""
Semantic ingestion tool for creating persistent databases from repositories.
This tool can be invoked via MCP to ingest a repository and create a database
that can later be used for semantic searches.
"""

import asyncio
import logging
import os
import time
import tempfile
from pathlib import Path
from typing import Optional

# Set HF_HOME to avoid transformers deprecation warning
if 'HF_HOME' not in os.environ:
    os.environ['HF_HOME'] = os.path.expanduser('~/.cache/huggingface')

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    from langchain_huggingface import HuggingFaceEmbeddings
    import numpy as np
except ImportError as e:
    raise ImportError(f"Missing required dependencies for semantic ingestion: {e}")

from ...models.base import BaseFrameworkModel
from ...models.tools import ToolExecutionContext
from ...models.semantic import (
    IndexingStats, EmbeddingModel, QdrantConfig, SemanticSearchConfig,
    CodeChunk, ChunkType
)
from ...models.semantic_ingestion import (
    SemanticIngestionRequest, SemanticIngestionResponse
)
from ...tool_registry import BaseTool
from .search import WorkingSemanticSearchTool


logger = logging.getLogger(__name__)


class SemanticIngestionTool(BaseTool):
    """
    Tool for ingesting repositories and creating persistent semantic databases.
    """
    
    def __init__(self):
        """Initialize the semantic ingestion tool."""
        self.temp_db_counter = 0
    
    @property
    def name(self) -> str:
        return "semantic_ingestion"
    
    @property
    def description(self) -> str:
        return "Ingest a repository folder and create a persistent semantic database for later querying"
    
    @property
    def input_schema(self):
        return SemanticIngestionRequest
    
    @property
    def output_schema(self):
        return SemanticIngestionResponse

    async def execute(
        self,
        input_data: SemanticIngestionRequest,
        context: ToolExecutionContext
    ) -> SemanticIngestionResponse:
        """Execute semantic ingestion with structured response."""
        start_time = time.time()
        
        try:
            logger.info(f"Starting semantic ingestion of repository: {input_data.repository_path}")
            
            # Determine database path
            if input_data.database_path:
                db_path = Path(input_data.database_path)
                db_path.mkdir(parents=True, exist_ok=True)
                db_location = str(db_path)
            else:
                # Create an ephemeral database in /dev/shm for fast access
                if Path("/dev/shm").exists():
                    # Use shared memory for ephemeral database
                    temp_dir = Path("/dev/shm") / "semantic_databases"
                    temp_dir.mkdir(exist_ok=True)
                    self.temp_db_counter += 1
                    db_name = f"ephemeral_semantic_{os.getpid()}_{int(time.time())}_{self.temp_db_counter}"
                    db_path = temp_dir / db_name
                    db_path.mkdir(parents=True, exist_ok=True)
                    db_location = str(db_path)
                    logger.info(f"Created ephemeral database in shared memory: {db_location}")
                else:
                    # Fallback to regular temp directory
                    temp_dir = Path(tempfile.gettempdir()) / "semantic_databases"
                    temp_dir.mkdir(exist_ok=True)
                    self.temp_db_counter += 1
                    db_name = f"ephemeral_semantic_{os.getpid()}_{int(time.time())}_{self.temp_db_counter}"
                    db_path = temp_dir / db_name
                    db_path.mkdir(parents=True, exist_ok=True)
                    db_location = str(db_path)
                    logger.info(f"Created ephemeral database in temp directory: {db_location}")
            
            logger.info(f"Using database location: {db_location}")
            
            # Configure embedding model
            embedding_model = self._get_embedding_model(input_data.embedding_model)
            
            # Configure Qdrant for persistent storage
            collection_name = f"repo_{Path(input_data.repository_path).name}_{int(time.time())}"
            qdrant_config = QdrantConfig(
                location=db_location,  # Persistent storage
                collection_name=collection_name,
                vector_size=embedding_model.dimension,
                prefer_grpc=False,
                timeout=60
            )
            
            # Create semantic search configuration
            config = SemanticSearchConfig(
                embedding_model=embedding_model,
                qdrant_config=qdrant_config,
                max_file_size=input_data.max_file_size,
                chunk_size=input_data.chunk_size,
                chunk_overlap=input_data.chunk_overlap
            )
            
            # Use the proper RAG ingestion logic instead of search tool
            from ...rag.ingestion import SemanticIngester
            
            logger.info("Starting repository indexing with proper RAG ingestion...")
            ingester = SemanticIngester()
            
            # Convert input data to the format expected by SemanticIngester
            ingestion_response = await ingester.ingest_repository(
                repository_path=input_data.repository_path,
                database_path=db_location,
                embedding_model=input_data.embedding_model,
                languages=input_data.languages,
                max_file_size=input_data.max_file_size,
                chunk_size=input_data.chunk_size,
                chunk_overlap=input_data.chunk_overlap
            )
            
            if not ingestion_response.success:
                raise Exception(ingestion_response.error_message or "Ingestion failed")
            
            indexing_stats = ingestion_response.indexing_stats
            
            # Save metadata about the ingestion for later querying
            metadata_file = Path(db_location) / "ingestion_metadata.json"
            try:
                import json
                metadata = {
                    "collection_name": collection_name,
                    "embedding_model": embedding_model.name,
                    "vector_dimension": embedding_model.dimension,
                    "repository_path": input_data.repository_path,
                    "ingestion_time": time.time(),
                    "languages": input_data.languages or [],
                    "files_processed": indexing_stats.files_processed,
                    "chunks_created": indexing_stats.chunks_created
                }
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                logger.info(f"Saved ingestion metadata to: {metadata_file}")
            except Exception as e:
                logger.warning(f"Could not save metadata: {e}")
            
            # Calculate database size
            db_size = self._calculate_directory_size(Path(db_location))
            
            # No cleanup needed since we're using the RAG ingester directly
            
            ingestion_time = time.time() - start_time
            logger.info(f"Ingestion completed in {ingestion_time:.2f} seconds")
            
            # Create successful response
            response = SemanticIngestionResponse(
                success=True,
                repository_path=input_data.repository_path,
                database_path=db_location,
                indexing_stats=indexing_stats,
                collection_name=collection_name,
                embedding_model_used=embedding_model.name,
                vector_dimension=embedding_model.dimension,
                total_database_size=db_size,
                ingestion_time=ingestion_time
            )
            
            logger.info(f"Successfully ingested repository with {indexing_stats.chunks_created} chunks")
            return response
            
        except Exception as e:
            error_msg = f"Semantic ingestion failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Return error response
            return SemanticIngestionResponse(
                success=False,
                repository_path=input_data.repository_path,
                database_path=db_location if 'db_location' in locals() else "",
                indexing_stats=IndexingStats(),  # Empty stats
                collection_name="",
                embedding_model_used="",
                vector_dimension=0,
                error_message=error_msg,
                ingestion_time=time.time() - start_time
            )

    def _get_embedding_model(self, model_name: str) -> EmbeddingModel:
        """Get embedding model configuration by name."""
        model_map = {
            "codebert": EmbeddingModel.codebert(),
            "unixcoder": EmbeddingModel.unixcoder(),
            "minilm": EmbeddingModel.minilm(),
            "codebert_st": EmbeddingModel.codebert_st()
        }
        
        if model_name not in model_map:
            logger.warning(f"Unknown embedding model '{model_name}', using default 'codebert_st'")
            model_name = "codebert_st"
        
        return model_map[model_name]

    def _calculate_directory_size(self, directory: Path) -> int:
        """Calculate total size of directory in bytes."""
        try:
            total_size = 0
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return total_size
        except Exception as e:
            logger.warning(f"Could not calculate directory size: {e}")
            return 0


class SemanticQueryTool(BaseTool):
    """
    Tool for querying previously ingested semantic databases.
    """
    
    @property
    def name(self) -> str:
        return "semantic_query"
    
    @property
    def description(self) -> str:
        return "Query a previously ingested semantic database with natural language"
    
    @property
    def input_schema(self):
        from ...models.semantic_ingestion import SemanticQueryRequest
        return SemanticQueryRequest
    
    @property
    def output_schema(self):
        from ...models.semantic import SemanticSearchResponse
        return SemanticSearchResponse

    async def execute(
        self,
        input_data,  # SemanticQueryRequest
        context: ToolExecutionContext
    ):
        """Execute semantic query against ingested database."""
        start_time = time.time()
        
        try:
            from ...models.semantic import SemanticSearchRequest, SemanticSearchResponse
            
            logger.info(f"Querying semantic database: {input_data.database_path}")
            
            # Find the collection in the database
            db_path = Path(input_data.database_path)
            if not db_path.exists():
                raise ValueError(f"Database path does not exist: {input_data.database_path}")
            
            # Try to load metadata first
            metadata_file = db_path / "ingestion_metadata.json"
            collection_name = None
            embedding_model_name = "codebert_st"
            
            if metadata_file.exists():
                try:
                    import json
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    collection_name = metadata.get("collection_name")
                    embedding_model_name = metadata.get("embedding_model", "codebert_st")
                    logger.info(f"Loaded metadata: collection={collection_name}, model={embedding_model_name}")
                except Exception as e:
                    logger.warning(f"Could not load metadata: {e}")
            
            # If no metadata, try to find the collection name from the database
            if not collection_name:
                try:
                    # Create a temporary client to inspect the database
                    temp_client = QdrantClient(location=str(db_path))
                    collections = temp_client.get_collections()
                    if collections.collections:
                        collection_name = collections.collections[0].name
                        logger.info(f"Found collection: {collection_name}")
                    else:
                        raise ValueError("No collections found in database")
                    temp_client.close()
                except Exception as e:
                    logger.error(f"Could not determine collection name: {e}")
                    # Try to guess the collection name based on the database path
                    db_name = db_path.name
                    if "semantic_db_" in db_name:
                        # Extract timestamp from database name and guess collection name
                        parts = db_name.split('_')
                        if len(parts) >= 3:
                            collection_name = f"repo_tmp{parts[-2]}_{parts[-1]}"
                        else:
                            collection_name = "code_chunks"
                    else:
                        collection_name = "code_chunks"  # Fallback
                    logger.info(f"Using guessed collection name: {collection_name}")
            
            # Configure for querying the persistent database
            # Use the same embedding model that was used for ingestion
            if embedding_model_name == "codebert":
                embedding_model = EmbeddingModel.codebert()
            elif embedding_model_name == "unixcoder":
                embedding_model = EmbeddingModel.unixcoder()
            elif embedding_model_name == "minilm":
                embedding_model = EmbeddingModel.minilm()
            else:
                embedding_model = EmbeddingModel.codebert_st()  # Default
            
            qdrant_config = QdrantConfig(
                location=str(db_path),
                collection_name=collection_name,
                vector_size=embedding_model.dimension
            )
            
            config = SemanticSearchConfig(
                embedding_model=embedding_model,
                qdrant_config=qdrant_config
            )
            
            # Create search request
            search_request = SemanticSearchRequest(
                query=input_data.query,
                directory="",  # Not used for querying existing DB
                languages=input_data.languages,
                file_patterns=input_data.file_patterns,
                max_results=input_data.max_results,
                similarity_threshold=input_data.similarity_threshold
            )
            
            # Initialize search tool and perform query
            search_tool = WorkingSemanticSearchTool(config)
            await search_tool.initialize()
            
            # Perform search
            response = await search_tool.execute(search_request, context)
            
            # Clean up
            await search_tool.close()
            
            logger.info(f"Query completed, found {response.total_results} results")
            return response
            
        except Exception as e:
            error_msg = f"Semantic query failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            from ...models.semantic import SemanticSearchResponse, SearchMode
            
            return SemanticSearchResponse(
                success=False,
                query=input_data.query,
                directory=input_data.database_path,
                results=[],
                total_results=0,
                search_time=time.time() - start_time,
                search_mode=SearchMode.SEMANTIC,
                similarity_threshold=input_data.similarity_threshold,
                error_message=error_msg
            )