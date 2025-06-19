"""
Semantic ingestion models for the LLM-MCP framework.
"""

from typing import List, Optional
from pydantic import Field, validator
from pathlib import Path

from .base import BaseFrameworkModel
from .semantic import IndexingStats, EmbeddingModel, QdrantConfig
from ..utils.constants import SUPPORTED_LANGUAGES, get_exclude_patterns_list


class SemanticIngestionRequest(BaseFrameworkModel):
    """Request model for semantic ingestion operations."""
    
    # Core parameters
    repository_path: str = Field(..., description="Path to the repository folder to ingest")
    database_path: Optional[str] = Field(None, description="Path for persistent database (if None, uses temp location)")
    
    # Language and file filtering
    languages: Optional[List[str]] = Field(None, description="Programming languages to include")
    file_patterns: Optional[List[str]] = Field(None, description="File patterns to include")
    exclude_patterns: List[str] = Field(
        default_factory=get_exclude_patterns_list,
        description="Patterns to exclude from ingestion"
    )
    
    # Processing configuration
    max_file_size: int = Field(1024 * 1024, description="Maximum file size in bytes")
    chunk_size: int = Field(1000, description="Size of code chunks")
    chunk_overlap: int = Field(200, description="Overlap between chunks")
    
    # Embedding configuration
    embedding_model: str = Field("codebert_st", description="Embedding model to use")
    force_reindex: bool = Field(False, description="Force re-indexing of all files")
    
    @validator('repository_path')
    def validate_repository_path(cls, v):
        """Validate that repository path exists."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Repository path does not exist: {v}")
        if not path.is_dir():
            raise ValueError(f"Repository path is not a directory: {v}")
        return str(path.resolve())
    
    @validator('languages')
    def validate_languages(cls, v):
        """Validate supported languages."""
        if v is None:
            return v
        
        invalid_languages = set(v) - SUPPORTED_LANGUAGES
        if invalid_languages:
            raise ValueError(f"Unsupported languages: {invalid_languages}")
        
        return v


class SemanticIngestionResponse(BaseFrameworkModel):
    """Response model for semantic ingestion operations."""
    
    success: bool = Field(..., description="Whether the ingestion was successful")
    repository_path: str = Field(..., description="Path to the ingested repository")
    database_path: str = Field(..., description="Path to the created database")
    
    # Ingestion statistics
    indexing_stats: IndexingStats = Field(..., description="Detailed indexing statistics")
    
    # Database information
    collection_name: str = Field(..., description="Name of the created collection")
    embedding_model_used: str = Field(..., description="Embedding model that was used")
    vector_dimension: int = Field(..., description="Dimension of the embedding vectors")
    
    # Error handling
    error_message: Optional[str] = Field(None, description="Error message if ingestion failed")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings during ingestion")
    
    # Metadata
    total_database_size: Optional[int] = Field(None, description="Total size of database in bytes")
    ingestion_time: float = Field(..., description="Total time taken for ingestion in seconds")


class SemanticQueryRequest(BaseFrameworkModel):
    """Request model for querying an ingested database."""
    
    query: str = Field(..., description="Natural language search query")
    database_path: str = Field(..., description="Path to the ingested database")
    
    # Search configuration
    max_results: int = Field(20, ge=1, le=100, description="Maximum number of results")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity score")
    
    # Filtering
    languages: Optional[List[str]] = Field(None, description="Filter by programming languages")
    file_patterns: Optional[List[str]] = Field(None, description="Filter by file patterns")
    
    @validator('database_path')
    def validate_database_path(cls, v):
        """Validate that database path exists."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Database path does not exist: {v}")
        return str(path.resolve())