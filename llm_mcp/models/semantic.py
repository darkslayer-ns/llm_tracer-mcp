"""
Semantic search models for the LLM-MCP framework.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import Field, validator

from .base import BaseFrameworkModel
from ..utils.constants import SUPPORTED_LANGUAGES, get_exclude_patterns_list, ALL_CODE_EXTENSIONS, DEFAULT_SIMILARITY_THRESHOLD


class SearchMode(str, Enum):
    """Search mode options."""
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    KEYWORD = "keyword"


class ChunkType(str, Enum):
    """Types of code chunks."""
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    DOCUMENTATION = "documentation"
    IMPORT = "import"
    VARIABLE = "variable"
    COMMENT = "comment"
    OTHER = "other"


class SemanticSearchRequest(BaseFrameworkModel):
    """Request model for semantic search operations."""
    
    # Core search parameters
    query: str = Field(..., description="Natural language search query")
    directory: str = Field(".", description="Directory to search in")
    
    # Language and file filtering
    languages: Optional[List[str]] = Field(None, description="Programming languages to include")
    file_patterns: Optional[List[str]] = Field(None, description="File patterns to include")
    exclude_patterns: Optional[List[str]] = Field(
        default_factory=get_exclude_patterns_list,
        description="Patterns to exclude"
    )
    
    # Search configuration
    max_results: int = Field(20, ge=1, le=100, description="Maximum number of results")
    similarity_threshold: float = Field(DEFAULT_SIMILARITY_THRESHOLD, ge=0.0, le=1.0, description="Minimum similarity score")
    include_context: bool = Field(True, description="Include surrounding code context")
    context_lines: int = Field(5, ge=0, le=20, description="Lines of context around matches")
    
    # Advanced options
    search_mode: SearchMode = Field(SearchMode.SEMANTIC, description="Search mode")
    chunk_types: Optional[List[ChunkType]] = Field(
        None,
        description="Types of code chunks to search (None = search all chunks)"
    )
    reindex: bool = Field(False, description="Force re-indexing of files")
    
    @validator('exclude_patterns')
    def validate_exclude_patterns(cls, v):
        """Validate exclude patterns, converting None to default list."""
        if v is None:
            return get_exclude_patterns_list()
        return v
    
    @validator('languages')
    def validate_languages(cls, v):
        """Validate supported languages."""
        if v is None:
            return v
        
        invalid_languages = set(v) - SUPPORTED_LANGUAGES
        if invalid_languages:
            raise ValueError(f"Unsupported languages: {invalid_languages}")
        
        return v


class CodeChunk(BaseFrameworkModel):
    """Represents a semantically meaningful code chunk."""
    
    # Content and location
    content: str = Field(..., description="Code chunk content")
    file_path: str = Field(..., description="Path to source file")
    start_line: int = Field(..., description="Starting line number")
    end_line: int = Field(..., description="Ending line number")
    
    # Semantic information
    chunk_type: ChunkType = Field(..., description="Type of code chunk")
    language: str = Field(..., description="Programming language")
    similarity_score: float = Field(..., description="Similarity score to query")
    
    # Code structure
    function_name: Optional[str] = Field(None, description="Function/method name if applicable")
    class_name: Optional[str] = Field(None, description="Class name if applicable")
    signature: Optional[str] = Field(None, description="Function/method signature")
    docstring: Optional[str] = Field(None, description="Associated documentation")
    
    # Context
    context_before: List[str] = Field(default_factory=list, description="Lines before the chunk")
    context_after: List[str] = Field(default_factory=list, description="Lines after the chunk")
    
    # Metadata
    imports: List[str] = Field(default_factory=list, description="Relevant imports/dependencies")
    tags: List[str] = Field(default_factory=list, description="Semantic tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class IndexingStats(BaseFrameworkModel):
    """Statistics about the indexing process."""
    
    files_processed: int = Field(0, description="Number of files processed")
    files_indexed: int = Field(0, description="Number of files successfully indexed")
    files_skipped: int = Field(0, description="Number of files skipped")
    chunks_created: int = Field(0, description="Number of code chunks created")
    languages_found: List[str] = Field(default_factory=list, description="Programming languages found")
    indexing_time: float = Field(0.0, description="Time taken for indexing in seconds")
    errors: List[str] = Field(default_factory=list, description="Indexing errors")


class SemanticSearchResponse(BaseFrameworkModel):
    """Structured response for semantic search operations."""
    
    success: bool = Field(..., description="Whether the operation was successful")
    query: str = Field(..., description="Original search query")
    directory: str = Field(..., description="Directory that was searched")
    
    # Results
    results: List[CodeChunk] = Field(default_factory=list, description="Search results")
    total_results: int = Field(0, description="Total number of results found")
    
    # Statistics
    indexing_stats: Optional[IndexingStats] = Field(None, description="Indexing statistics")
    search_time: float = Field(0.0, description="Time taken for search in seconds")
    
    # Configuration used
    search_mode: SearchMode = Field(..., description="Search mode used")
    similarity_threshold: float = Field(..., description="Similarity threshold used")
    
    # Error handling
    error_message: Optional[str] = Field(None, description="Error message if operation failed")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings")


class EmbeddingModel(BaseFrameworkModel):
    """Configuration for embedding models."""
    
    name: str = Field(..., description="Model name")
    model_path: str = Field(..., description="Path or identifier for the model")
    dimension: int = Field(..., description="Embedding dimension")
    max_length: int = Field(512, description="Maximum input length")
    device: str = Field("cpu", description="Device to run on")
    
    # Predefined models
    @classmethod
    def codebert(cls) -> "EmbeddingModel":
        """CodeBERT model optimized for code understanding."""
        return cls(
            name="codebert",
            model_path="microsoft/codebert-base",
            dimension=768,
            max_length=512
        )
    
    @classmethod
    def unixcoder(cls) -> "EmbeddingModel":
        """UniXcoder model for multi-language code understanding."""
        return cls(
            name="unixcoder",
            model_path="microsoft/unixcoder-base",
            dimension=768,
            max_length=512
        )
    
    @classmethod
    def minilm(cls) -> "EmbeddingModel":
        """Lightweight MiniLM model for general use."""
        return cls(
            name="minilm",
            model_path="sentence-transformers/all-MiniLM-L6-v2",
            dimension=384,
            max_length=256
        )
    
    @classmethod
    def codebert_st(cls) -> "EmbeddingModel":
        """CodeBERT model via sentence-transformers for better code understanding."""
        return cls(
            name="codebert_st",
            model_path="sentence-transformers/all-mpnet-base-v2",  # High-quality model good for code
            dimension=768,
            max_length=384
        )


class QdrantConfig(BaseFrameworkModel):
    """Configuration for Qdrant vector database."""
    
    # Database settings
    location: str = Field(":memory:", description="Database location")
    prefer_grpc: bool = Field(False, description="Use gRPC instead of HTTP")
    timeout: int = Field(30, description="Connection timeout in seconds")
    
    # Collection settings
    collection_name: str = Field("code_chunks", description="Collection name")
    vector_size: int = Field(384, description="Vector dimension")
    distance_metric: str = Field("Cosine", description="Distance metric")
    
    # Performance settings
    segment_number: int = Field(2, description="Number of segments")
    max_segment_size: int = Field(100000, description="Maximum segment size")
    memmap_threshold: int = Field(20000, description="Memory mapping threshold")
    indexing_threshold: int = Field(10000, description="Indexing threshold")
    
    # HNSW settings
    hnsw_m: int = Field(16, description="HNSW M parameter")
    hnsw_ef_construct: int = Field(100, description="HNSW ef_construct parameter")
    full_scan_threshold: int = Field(10000, description="Full scan threshold")


class SemanticSearchConfig(BaseFrameworkModel):
    """Configuration for semantic search tool."""
    
    embedding_model: EmbeddingModel = Field(default_factory=EmbeddingModel.minilm)
    qdrant_config: QdrantConfig = Field(default_factory=QdrantConfig)
    
    # Processing settings
    max_file_size: int = Field(1024 * 1024, description="Maximum file size in bytes")
    chunk_size: int = Field(1000, description="Default chunk size")
    chunk_overlap: int = Field(200, description="Chunk overlap size")
    
    # Performance settings
    batch_size: int = Field(32, description="Batch size for processing")
    max_workers: int = Field(4, description="Maximum worker threads")
    cache_embeddings: bool = Field(True, description="Cache embeddings")
    
    # Language settings
    default_languages: List[str] = Field(
        default_factory=lambda: list(SUPPORTED_LANGUAGES),
        description="Default languages to process"
    )