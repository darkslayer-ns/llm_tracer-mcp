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

    async def index_directory(self, directory: str) -> IndexingStats:
        """Index all files in directory first, before any searches."""
        try:
            await self.initialize()
            
            # Create a dummy request for indexing
            dummy_request = SemanticSearchRequest(
                query="",  # Not used for indexing
                directory=directory
            )
            
            # Index all files
            stats = await self._index_directory_incremental(dummy_request)
            logger.info(f"Pre-indexed directory: {stats.files_processed} files, {stats.chunks_created} chunks")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to index directory: {e}")
            raise

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
            
            # Handle reindex parameter - force indexing if requested
            indexing_stats = None
            if input_data.reindex:
                logger.info("Reindex requested - performing incremental indexing")
                indexing_stats = await self._index_directory_incremental(input_data)
            else:
                # Skip indexing during search - assume it's already done
                # If you want to force indexing, call index_directory() first
                pass
            
            # Search for similar chunks
            results = await self._search_similar(input_data)
            
            # Use actual indexing stats if reindexing was performed, otherwise empty stats
            if indexing_stats is None:
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

    async def _index_directory_incremental(self, request: SemanticSearchRequest) -> IndexingStats:
        """Index files incrementally - only new/changed files."""
        start_time = time.time()
        total_files = 0
        total_chunks = 0
        new_files = 0
        
        # Use directory field from SemanticSearchRequest
        path = Path(request.directory)
        if not path.exists():
            return IndexingStats(
                files_processed=0,
                files_indexed=0,
                files_skipped=0,
                chunks_created=0,
                languages_found=[],
                indexing_time=0.0,
                errors=[]
            )
        
        if path.is_file():
            files_to_process = [path]
        else:
            # Get all files in directory
            exclude_patterns = self._get_comprehensive_exclude_patterns(request.exclude_patterns or [])
            suffixes = self._get_file_suffixes(request)
            
            files_to_process = []
            for suffix in suffixes:
                pattern = f"**/*{suffix}"
                for file_path in path.glob(pattern):
                    if file_path.is_file() and not self._should_exclude_file(file_path, exclude_patterns):
                        files_to_process.append(file_path)
        
        # Filter to only new/changed files (or all files if reindex=True)
        files_to_index = []
        for file_path in files_to_process:
            file_str = str(file_path)
            try:
                current_mtime = file_path.stat().st_mtime
                
                # If reindex is requested, index all files regardless of cache
                if request.reindex or file_str not in self.indexed_files or self.indexed_files[file_str] != current_mtime:
                    files_to_index.append(file_path)
                    self.indexed_files[file_str] = current_mtime
                    new_files += 1
            except Exception as e:
                logger.warning(f"Error checking file {file_path}: {e}")
        
        if files_to_index:
            # Process new/changed files
            chunks = await self._process_files(files_to_index, request)
            if chunks:
                await self._index_chunks(chunks)
                total_chunks += len(chunks)
            
            total_files += len(files_to_index)
        
        # Save updated cache
        await self._save_index_cache()
        
        indexing_time = time.time() - start_time
        logger.info(f"Indexed {new_files} new/changed files, {total_chunks} chunks")
        
        return IndexingStats(
            files_processed=total_files,
            files_indexed=new_files,
            files_skipped=len(files_to_process) - new_files,
            chunks_created=total_chunks,
            languages_found=list(set()),
            indexing_time=indexing_time,
            errors=[]
        )

    async def _process_files(self, files: List[Path], request: SemanticSearchRequest) -> List[CodeChunk]:
        """Process files using LangChain LanguageParser."""
        chunks = []
        
        for file_path in files:
            try:
                # Simple text processing for now to avoid langchain issues
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                
                # Create a simple chunk for the whole file
                chunk = CodeChunk(
                    content=content[:1000],  # Limit content size
                    file_path=str(file_path),
                    start_line=1,
                    end_line=len(content.split('\n')),
                    language=self._detect_language_from_path(str(file_path)),
                    chunk_type=ChunkType.OTHER,
                    similarity_score=0.0,  # Use correct field name
                    function_name=None,
                    class_name=None,
                    signature=None
                )
                chunks.append(chunk)
                        
            except Exception as e:
                logger.warning(f"Error processing file {file_path}: {e}")
                continue
        
        return chunks

    def _detect_language_from_path(self, file_path: str) -> str:
        """Detect programming language from file path."""
        path = Path(file_path)
        suffix = path.suffix.lower()
        
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.rb': 'ruby'
        }
        
        return language_map.get(suffix, 'unknown')

    async def _index_chunks(self, chunks: List[CodeChunk]) -> None:
        """Index code chunks into Qdrant using neural embeddings."""
        try:
            # Prepare texts for embedding
            texts = [self._prepare_text_for_embedding(chunk) for chunk in chunks]
            
            # Generate embeddings using LangChain HuggingFace embeddings
            embeddings = self.embeddings.embed_documents(texts)
            
            # Ensure embeddings are numpy arrays
            embeddings = np.array(embeddings)
            
            # Create points for Qdrant
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point = PointStruct(
                    id=hash(f"{chunk.file_path}:{chunk.start_line}:{chunk.content[:50]}"),
                    vector=embedding.tolist(),
                    payload={
                        "content": chunk.content,
                        "file_path": chunk.file_path,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "language": chunk.language,
                        "chunk_type": chunk.chunk_type.value if hasattr(chunk.chunk_type, 'value') else str(chunk.chunk_type),
                        "function_name": chunk.function_name,
                        "class_name": chunk.class_name,
                        "signature": chunk.signature
                    }
                )
                points.append(point)
            
            # Upload to Qdrant
            self.qdrant_client.upsert(
                collection_name=self.config.qdrant_config.collection_name,
                points=points
            )
            
            logger.info(f"Indexed {len(chunks)} chunks into Qdrant")
            
        except Exception as e:
            logger.error(f"Failed to index chunks: {e}")
            raise

    def _prepare_text_for_embedding(self, chunk: CodeChunk) -> str:
        """Prepare text for embedding by combining relevant fields."""
        parts = [chunk.content]
        
        if chunk.function_name:
            parts.append(f"function {chunk.function_name}")
        
        if chunk.class_name:
            parts.append(f"class {chunk.class_name}")
        
        # Add language and chunk type as context
        parts.append(f"language {chunk.language}")
        parts.append(f"type {chunk.chunk_type.value if hasattr(chunk.chunk_type, 'value') else str(chunk.chunk_type)}")
        
        return " ".join(parts)

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

    def _get_comprehensive_exclude_patterns(self, user_excludes: List[str]) -> List[str]:
        """Get comprehensive exclude patterns including common cache/build directories."""
        default_excludes = [
            "node_modules", ".git", ".svn", ".hg", "__pycache__", ".pytest_cache",
            "venv", "env", ".venv", ".env", "virtualenv",
            "build", "dist", "target", "bin", "obj",
            ".idea", ".vscode", ".vs", "*.pyc", "*.pyo", "*.pyd",
            ".DS_Store", "Thumbs.db", "*.log", "*.tmp", "*.temp"
        ]
        return default_excludes + (user_excludes or [])

    def _get_file_suffixes(self, request: SemanticSearchRequest) -> List[str]:
        """Get file suffixes based on request parameters."""
        if hasattr(request, 'languages') and request.languages:
            # Map languages to file extensions
            lang_map = {
                'python': '.py', 'javascript': '.js', 'typescript': '.ts',
                'java': '.java', 'cpp': '.cpp', 'c': '.c', 'go': '.go',
                'rust': '.rs', 'php': '.php', 'ruby': '.rb'
            }
            return [lang_map.get(lang, f".{lang}") for lang in request.languages]
        
        # Default to common programming language extensions
        return [
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.hpp',
            '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.scala', '.r',
            '.m', '.sh', '.sql', '.html', '.css', '.scss', '.less', '.xml',
            '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf',
            '.md', '.rst', '.txt'
        ]

    def _should_exclude_file(self, file_path: Path, exclude_patterns: List[str]) -> bool:
        """Check if file should be excluded based on patterns."""
        file_str = str(file_path)
        file_name = file_path.name
        
        for pattern in exclude_patterns:
            if pattern in file_str or pattern in file_name:
                return True
            # Handle glob-like patterns
            if pattern.startswith('*.') and file_name.endswith(pattern[1:]):
                return True
        
        return False

    async def close(self) -> None:
        """Clean up resources."""
        try:
            await self._save_index_cache()
            if self.qdrant_client:
                self.qdrant_client.close()
            logger.info("Working semantic search tool closed")
        except Exception as e:
            logger.warning(f"Error closing semantic search tool: {e}")