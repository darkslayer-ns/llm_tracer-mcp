"""
Working semantic search tool with proper incremental indexing.
Uses LangChain HuggingFace embeddings for superior code understanding.
"""

import asyncio
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Set HF_HOME to avoid transformers deprecation warning
if 'HF_HOME' not in os.environ:
    os.environ['HF_HOME'] = os.path.expanduser('~/.cache/huggingface')

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    from langchain_community.document_loaders.generic import GenericLoader
    from langchain_community.document_loaders.parsers import LanguageParser
    from langchain.schema import Document
    from langchain_huggingface import HuggingFaceEmbeddings
    import numpy as np
except ImportError as e:
    raise ImportError(f"Missing required dependencies for semantic search: {e}")

from ...models.base import BaseFrameworkModel
from ...models.tools import ToolExecutionContext
from ...models.semantic import (
    SemanticSearchRequest, SemanticSearchResponse, CodeChunk, ChunkType,
    IndexingStats, EmbeddingModel, QdrantConfig, SemanticSearchConfig
)
from ...tool_registry import BaseTool


logger = logging.getLogger(__name__)


class WorkingSemanticSearchTool(BaseTool):
    """
    Working semantic search tool with proper incremental indexing.
    """
    
    def __init__(self, config: Optional[SemanticSearchConfig] = None):
        """Initialize the semantic search tool."""
        self.config = config or SemanticSearchConfig()
        self.qdrant_client: Optional[QdrantClient] = None
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self.indexed_files: Dict[str, float] = {}  # file_path -> last_modified
        self.is_initialized = False
        self.index_cache_path = Path("/tmp/working_search_index.pkl")
    
    @property
    def name(self) -> str:
        return "working_semantic_search"
    
    @property
    def description(self) -> str:
        return "Working semantic search across code files with proper incremental indexing"
    
    @property
    def input_schema(self):
        return SemanticSearchRequest
    
    @property
    def output_schema(self):
        return SemanticSearchResponse

    async def initialize(self) -> None:
        """Initialize the semantic search components."""
        if self.is_initialized:
            return
        
        try:
            # Initialize Qdrant client (use configured location)
            qdrant_location = self.config.qdrant_config.location
            if qdrant_location == ":memory:":
                self.qdrant_client = QdrantClient(":memory:")
            else:
                # Ensure the directory exists for persistent storage
                from pathlib import Path
                if qdrant_location != ":memory:":
                    Path(qdrant_location).mkdir(parents=True, exist_ok=True)
                self.qdrant_client = QdrantClient(path=qdrant_location)
            logger.info(f"Initialized Qdrant client with location: {qdrant_location}")
            
            # Initialize LangChain HuggingFace embeddings with the configured model
            model_name = self.config.embedding_model.model_path
            logger.info(f"Loading LangChain HuggingFace embeddings: {model_name}")
            
            # Set up model caching via environment variables (standard HuggingFace approach)
            # Models will be cached in ~/.cache/huggingface/ to avoid re-downloading
            import os
            cache_dir = os.path.expanduser("~/.cache/huggingface")
            os.makedirs(cache_dir, exist_ok=True)
            os.environ['HF_HOME'] = cache_dir
            # os.environ['HF_HOME'] = os.path.join(cache_dir, 'transformers')
            logger.info(f"Using model cache directory: {cache_dir}")
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},  # Use CPU to avoid CUDA issues
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Ensure collection exists with correct vector size
            await self._ensure_collection_exists()
            
            # Try to load existing index
            await self._load_index_cache()
            
            self.is_initialized = True
            logger.info("Working semantic search tool initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize semantic search tool: {e}")
            raise

    # Removed index_directory method - search tool should only search existing databases

    async def execute(
        self,
        input_data: SemanticSearchRequest,
        context: ToolExecutionContext
    ) -> SemanticSearchResponse:
        """Execute semantic search with structured response."""
        start_time = time.time()
        
        try:
            # Initialize if needed
            await self.initialize()
            
            # Search for similar chunks (no indexing in search tool)
            results = await self._search_similar(input_data)
            
            # Empty indexing stats since search tool doesn't index
            indexing_stats = IndexingStats(
                files_processed=0,
                files_indexed=0,
                files_skipped=0,
                chunks_created=0,
                languages_found=[],
                indexing_time=0.0,
                errors=[]
            )
            
            # Build response
            response = SemanticSearchResponse(
                success=True,
                query=input_data.query,
                directory=input_data.directory,
                results=results,
                total_results=len(results),
                search_time=time.time() - start_time,
                search_mode=input_data.search_mode,
                similarity_threshold=input_data.similarity_threshold,
                indexing_stats=indexing_stats
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            raise

    async def _ensure_collection_exists(self) -> None:
        """Ensure the Qdrant collection exists."""
        collection_name = self.config.qdrant_config.collection_name
        
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if collection_name not in collection_names:
                # Create collection with correct vector size
                vector_size = self.config.embedding_model.dimension
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {e}")
            raise

    async def _load_index_cache(self) -> None:
        """Load previously indexed files from cache."""
        try:
            if self.index_cache_path.exists():
                with open(self.index_cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.indexed_files = cache_data.get('indexed_files', {})
                    # Note: We don't restore vectorizer state as it needs to be re-fitted
                logger.info(f"Loaded index cache with {len(self.indexed_files)} files")
        except Exception as e:
            logger.warning(f"Failed to load index cache: {e}")
            self.indexed_files = {}

    async def _save_index_cache(self) -> None:
        """Save indexed files to cache."""
        try:
            cache_data = {
                'indexed_files': self.indexed_files
            }
            with open(self.index_cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            logger.warning(f"Failed to save index cache: {e}")

    # Removed all indexing methods - search tool should only search existing databases

    async def _search_similar(self, request: SemanticSearchRequest) -> List[CodeChunk]:
        """Search for similar code chunks using neural embeddings."""
        try:
            # Check if embeddings are available
            if not self.embeddings:
                # If no embeddings loaded yet, return empty results
                return []
            
            # Generate query embedding using LangChain HuggingFace embeddings
            query_embedding = self.embeddings.embed_query(request.query)
            
            # Ensure query embedding is numpy array
            query_embedding = np.array(query_embedding)
            
            # Build search filter if chunk types are specified
            from qdrant_client.models import Filter, FieldCondition, MatchAny
            
            search_filter = None
            if hasattr(request, 'chunk_types') and request.chunk_types:
                chunk_type_values = [ct.value if hasattr(ct, 'value') else str(ct) for ct in request.chunk_types]
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="chunk_type",
                            match=MatchAny(any=chunk_type_values)
                        )
                    ]
                )
                logger.info(f"Using chunk type filter: {chunk_type_values}")
            else:
                logger.info("No chunk type filter applied")
            
            # Search in Qdrant
            logger.info(f"Searching in collection: {self.config.qdrant_config.collection_name}")
            logger.info(f"Search params: limit={request.max_results}, threshold={request.similarity_threshold}")
            
            # Verify collection exists and has data
            try:
                collection_info = self.qdrant_client.get_collection(self.config.qdrant_config.collection_name)
                logger.debug(f"Searching in collection with {collection_info.points_count} points")
            except Exception as e:
                logger.warning(f"Could not get collection info: {e}")
            
            # Perform semantic search in Qdrant using new query_points API
            search_results = self.qdrant_client.query_points(
                collection_name=self.config.qdrant_config.collection_name,
                query=query_embedding.tolist(),
                query_filter=search_filter,
                limit=request.max_results,
                score_threshold=request.similarity_threshold if request.similarity_threshold > 0.001 else None
            ).points
            logger.info(f"Found {len(search_results)} search results")
            
            # Convert results to CodeChunk objects
            chunks = []
            for result in search_results:
                payload = result.payload
                logger.debug(f"Processing result with score {result.score}: {payload.get('file_path', 'unknown')}")
                
                # Handle chunk_type conversion more safely
                chunk_type_value = payload.get("chunk_type", "OTHER")
                try:
                    if isinstance(chunk_type_value, str):
                        chunk_type = ChunkType(chunk_type_value)
                    else:
                        chunk_type = ChunkType.OTHER
                except ValueError:
                    logger.warning(f"Unknown chunk_type: {chunk_type_value}, using OTHER")
                    chunk_type = ChunkType.OTHER
                
                chunk = CodeChunk(
                    content=payload["content"],
                    file_path=payload["file_path"],
                    start_line=payload["start_line"],
                    end_line=payload["end_line"],
                    language=payload["language"],
                    chunk_type=chunk_type,
                    function_name=payload.get("function_name"),
                    class_name=payload.get("class_name"),
                    signature=payload.get("signature"),
                    similarity_score=result.score  # Use similarity_score instead of score
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    # Removed exclusion pattern methods - search tool should only search existing databases

    async def close(self) -> None:
        """Clean up resources."""
        try:
            await self._save_index_cache()
            if self.qdrant_client:
                self.qdrant_client.close()
            logger.info("Working semantic search tool closed")
        except Exception as e:
            logger.warning(f"Error closing semantic search tool: {e}")