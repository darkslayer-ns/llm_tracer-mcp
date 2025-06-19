"""
Semantic ingestion functionality for creating databases from repositories.
This module provides the core ingestion logic using LangChain's LanguageParser.
"""

import asyncio
import logging
import os
import time
import tempfile
from pathlib import Path
from typing import Optional, List

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
    raise ImportError(f"Missing required dependencies for semantic ingestion: {e}")

from ..models.semantic import (
    IndexingStats, EmbeddingModel, QdrantConfig, SemanticSearchConfig,
    CodeChunk, ChunkType
)
from ..models.semantic_ingestion import (
    SemanticIngestionRequest, SemanticIngestionResponse
)
from ..utils.constants import (
    LANGUAGE_EXTENSIONS, SUPPORTED_LANGUAGES, ALL_CODE_EXTENSIONS,
    EXCLUDED_PATTERNS, MAX_FILE_SIZE_BYTES, DEFAULT_CHUNK_SIZE, 
    DEFAULT_CHUNK_OVERLAP, EMBEDDING_MODELS, DEFAULT_EMBEDDING_MODEL,
    get_language_from_extension, get_extensions_for_language,
    should_exclude_path, get_exclude_patterns_list
)


logger = logging.getLogger(__name__)


class SemanticIngester:
    """
    Core semantic ingestion functionality for creating databases from repositories.
    Uses LangChain's LanguageParser for proper code parsing and chunking.
    """
    
    def __init__(self):
        """Initialize the semantic ingester."""
        self.temp_db_counter = 0
        self.qdrant_client: Optional[QdrantClient] = None
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
    
    async def ingest_repository(
        self,
        repository_path: str,
        database_path: Optional[str] = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        languages: Optional[list] = None,
        max_file_size: int = MAX_FILE_SIZE_BYTES,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    ) -> SemanticIngestionResponse:
        """
        Ingest a repository and create a semantic database using LangChain LanguageParser.
        
        Args:
            repository_path: Path to repository to ingest
            database_path: Path for database (None for ephemeral)
            embedding_model: Embedding model to use
            languages: Programming languages to include
            max_file_size: Maximum file size to process
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            
        Returns:
            SemanticIngestionResponse with ingestion results
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting semantic ingestion of repository: {repository_path}")
            
            # Determine database path
            if database_path:
                db_path = Path(database_path)
                db_path.mkdir(parents=True, exist_ok=True)
                db_location = str(db_path)
            else:
                # Create an ephemeral database in /dev/shm for fast access
                db_location = self._create_ephemeral_db_path()
            
            logger.info(f"Using database location: {db_location}")
            
            # Configure embedding model
            embedding_model_config = self._get_embedding_model(embedding_model)
            
            # Configure Qdrant for storage
            collection_name = f"repo_{Path(repository_path).name}_{int(time.time())}"
            qdrant_config = QdrantConfig(
                location=db_location,
                collection_name=collection_name,
                vector_size=embedding_model_config.dimension,
                prefer_grpc=False,
                timeout=60
            )
            
            # Initialize Qdrant client
            await self._initialize_qdrant(qdrant_config)
            
            # Initialize embeddings
            await self._initialize_embeddings(embedding_model_config)
            
            # Process files using LangChain LanguageParser
            logger.info("Processing files with LangChain LanguageParser...")
            documents = await self._process_repository_with_langchain(
                repository_path, languages, max_file_size
            )
            
            # Convert documents to code chunks
            logger.info(f"Converting {len(documents)} documents to code chunks...")
            chunks = await self._documents_to_chunks(documents)
            
            # Index chunks into Qdrant
            logger.info(f"Indexing {len(chunks)} chunks into Qdrant...")
            await self._index_chunks(chunks, qdrant_config.collection_name)
            
            # Create indexing stats
            indexing_stats = IndexingStats(
                files_processed=len(set(doc.metadata.get('source', '') for doc in documents)),
                files_indexed=len(set(doc.metadata.get('source', '') for doc in documents)),
                files_skipped=0,
                chunks_created=len(chunks),
                languages_found=list(set(chunk.language for chunk in chunks)),
                indexing_time=time.time() - start_time,
                errors=[]
            )
            
            # Save metadata about the ingestion for later querying
            self._save_metadata(
                db_location, collection_name, embedding_model_config,
                repository_path, languages, indexing_stats
            )
            
            # Calculate database size
            db_size = self._calculate_directory_size(Path(db_location))
            
            # Clean up
            await self._cleanup()
            
            ingestion_time = time.time() - start_time
            logger.info(f"Ingestion completed in {ingestion_time:.2f} seconds")
            
            # Create successful response
            response = SemanticIngestionResponse(
                success=True,
                repository_path=repository_path,
                database_path=db_location,
                indexing_stats=indexing_stats,
                collection_name=collection_name,
                embedding_model_used=embedding_model_config.name,
                vector_dimension=embedding_model_config.dimension,
                total_database_size=db_size,
                ingestion_time=ingestion_time
            )
            
            logger.info(f"Successfully ingested repository with {indexing_stats.chunks_created} chunks")
            return response
            
        except Exception as e:
            error_msg = f"Semantic ingestion failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Clean up on error
            await self._cleanup()
            
            # Return error response
            return SemanticIngestionResponse(
                success=False,
                repository_path=repository_path,
                database_path=db_location if 'db_location' in locals() else "",
                indexing_stats=IndexingStats(),  # Empty stats
                collection_name="",
                embedding_model_used="",
                vector_dimension=0,
                error_message=error_msg,
                ingestion_time=time.time() - start_time
            )
    
    async def _process_repository_with_langchain(
        self, 
        repository_path: str, 
        languages: Optional[list], 
        max_file_size: int
    ) -> List[Document]:
        """Process repository using LangChain's LanguageParser."""
        try:
            repo_path = Path(repository_path)
            if not repo_path.exists():
                raise ValueError(f"Repository path does not exist: {repository_path}")
            
            # Get file suffixes based on languages using constants
            suffixes = self._get_file_suffixes_from_languages(languages)
            
            # Pre-filter files to avoid processing excluded ones
            valid_files = []
            for suffix in suffixes:
                for file_path in repo_path.rglob(f"*{suffix}"):
                    if file_path.is_file():
                        # Check exclusion patterns first
                        if should_exclude_path(file_path.parts):
                            logger.debug(f"Excluding file due to pattern: {file_path}")
                            continue
                        
                        # Check file size
                        if file_path.stat().st_size <= max_file_size:
                            valid_files.append(file_path)
                        else:
                            logger.debug(f"Skipping large file: {file_path}")
            
            logger.info(f"Found {len(valid_files)} valid files to process")
            
            # Process only valid files with LangChain
            filtered_docs = []
            for file_path in valid_files:
                try:
                    # Use LangChain GenericLoader for individual files
                    loader = GenericLoader.from_filesystem(
                        path=str(file_path.parent),
                        glob=file_path.name,
                        parser=LanguageParser(),
                        show_progress=False
                    )
                    
                    # Load and process this file
                    file_docs = loader.load()
                    for doc in file_docs:
                        # Add language metadata using constants
                        doc.metadata['language'] = get_language_from_extension(file_path.suffix)
                        filtered_docs.append(doc)
                        
                except Exception as e:
                    logger.warning(f"Failed to process file {file_path} with LangChain: {e}")
                    # Fallback to simple text reading
                    try:
                        content = file_path.read_text(encoding='utf-8', errors='ignore')
                        doc = Document(
                            page_content=content,
                            metadata={
                                'source': str(file_path),
                                'language': get_language_from_extension(file_path.suffix)
                            }
                        )
                        filtered_docs.append(doc)
                    except Exception as e2:
                        logger.warning(f"Failed to read file {file_path}: {e2}")
            
            logger.info(f"Loaded {len(filtered_docs)} documents from {len(valid_files)} total files")
            return filtered_docs
            
        except Exception as e:
            logger.error(f"Failed to process repository with LangChain: {e}")
            # Fallback to simple file processing
            return await self._fallback_file_processing(repository_path, languages, max_file_size)
    
    async def _fallback_file_processing(
        self, 
        repository_path: str, 
        languages: Optional[list], 
        max_file_size: int
    ) -> List[Document]:
        """Fallback file processing if LangChain fails."""
        logger.info("Using fallback file processing...")
        documents = []
        repo_path = Path(repository_path)
        suffixes = self._get_file_suffixes_from_languages(languages)
        
        for suffix in suffixes:
            for file_path in repo_path.rglob(f"*{suffix}"):
                if file_path.is_file():
                    # Check exclusion patterns
                    if should_exclude_path(file_path.parts):
                        continue
                    
                    # Check file size
                    if file_path.stat().st_size <= max_file_size:
                        try:
                            content = file_path.read_text(encoding='utf-8', errors='ignore')
                            doc = Document(
                                page_content=content,
                                metadata={
                                    'source': str(file_path),
                                    'language': get_language_from_extension(file_path.suffix)
                                }
                            )
                            documents.append(doc)
                        except Exception as e:
                            logger.warning(f"Failed to read file {file_path}: {e}")
        
        return documents
    
    async def _documents_to_chunks(self, documents: List[Document]) -> List[CodeChunk]:
        """Convert LangChain documents to CodeChunk objects."""
        chunks = []
        
        for doc in documents:
            try:
                # Extract metadata
                source = doc.metadata.get('source', 'unknown')
                language = doc.metadata.get('language', 'unknown')
                
                # For now, create one chunk per document
                # TODO: Implement proper chunking based on code structure
                chunk = CodeChunk(
                    content=doc.page_content[:2000],  # Limit content size
                    file_path=source,
                    start_line=1,
                    end_line=len(doc.page_content.split('\n')),
                    language=language,
                    chunk_type=ChunkType.OTHER,  # TODO: Detect chunk type from content
                    similarity_score=0.0,
                    function_name=None,  # TODO: Extract from parsed content
                    class_name=None,     # TODO: Extract from parsed content
                    signature=None       # TODO: Extract from parsed content
                )
                chunks.append(chunk)
                
            except Exception as e:
                logger.warning(f"Failed to process document {doc.metadata.get('source', 'unknown')}: {e}")
        
        return chunks
    
    async def _initialize_qdrant(self, qdrant_config: QdrantConfig):
        """Initialize Qdrant client and collection."""
        try:
            # Initialize Qdrant client
            if qdrant_config.location == ":memory:":
                self.qdrant_client = QdrantClient(":memory:")
            else:
                Path(qdrant_config.location).mkdir(parents=True, exist_ok=True)
                self.qdrant_client = QdrantClient(path=qdrant_config.location)
            
            # Create collection
            collections = self.qdrant_client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if qdrant_config.collection_name not in collection_names:
                self.qdrant_client.create_collection(
                    collection_name=qdrant_config.collection_name,
                    vectors_config=VectorParams(
                        size=qdrant_config.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {qdrant_config.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            raise
    
    async def _initialize_embeddings(self, embedding_model: EmbeddingModel):
        """Initialize HuggingFace embeddings."""
        try:
            # Set up model caching
            cache_dir = os.path.expanduser("~/.cache/huggingface")
            os.makedirs(cache_dir, exist_ok=True)
            os.environ['HF_HOME'] = cache_dir
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model.model_path,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            logger.info(f"Initialized embeddings: {embedding_model.name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise
    
    async def _index_chunks(self, chunks: List[CodeChunk], collection_name: str):
        """Index code chunks into Qdrant."""
        try:
            # Prepare texts for embedding
            texts = [self._prepare_text_for_embedding(chunk) for chunk in chunks]
            
            # Generate embeddings
            embeddings = self.embeddings.embed_documents(texts)
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
                collection_name=collection_name,
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
    
    def _create_ephemeral_db_path(self) -> str:
        """Create an ephemeral database path."""
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
        
        return db_location
    
    def _get_embedding_model(self, model_name: str) -> EmbeddingModel:
        """Get embedding model configuration by name using constants."""
        if model_name not in EMBEDDING_MODELS:
            logger.warning(f"Unknown embedding model '{model_name}', using default '{DEFAULT_EMBEDDING_MODEL}'")
            model_name = DEFAULT_EMBEDDING_MODEL
        
        model_config = EMBEDDING_MODELS[model_name]
        return EmbeddingModel(
            name=model_name,
            model_path=model_config["model_path"],
            dimension=model_config["dimension"],
            max_length=model_config["max_length"]
        )
    
    def _get_file_suffixes_from_languages(self, languages: Optional[list]) -> List[str]:
        """Get file suffixes based on languages using constants."""
        if languages:
            # Get extensions for specified languages
            suffixes = []
            for lang in languages:
                if lang.lower() in LANGUAGE_EXTENSIONS:
                    suffixes.extend(LANGUAGE_EXTENSIONS[lang.lower()])
                else:
                    logger.warning(f"Unknown language: {lang}")
            return suffixes if suffixes else ALL_CODE_EXTENSIONS
        
        # Return all supported extensions
        return ALL_CODE_EXTENSIONS
    
    def _save_metadata(
        self, 
        db_location: str, 
        collection_name: str, 
        embedding_model: EmbeddingModel,
        repository_path: str, 
        languages: Optional[list], 
        indexing_stats: IndexingStats
    ):
        """Save metadata about the ingestion for later querying."""
        metadata_file = Path(db_location) / "ingestion_metadata.json"
        try:
            import json
            metadata = {
                "collection_name": collection_name,
                "embedding_model": embedding_model.name,
                "vector_dimension": embedding_model.dimension,
                "repository_path": repository_path,
                "ingestion_time": time.time(),
                "languages": languages or [],
                "files_processed": indexing_stats.files_processed,
                "chunks_created": indexing_stats.chunks_created
            }
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved ingestion metadata to: {metadata_file}")
        except Exception as e:
            logger.warning(f"Could not save metadata: {e}")
    
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
    
    async def _cleanup(self):
        """Clean up resources."""
        try:
            if self.qdrant_client:
                self.qdrant_client.close()
                self.qdrant_client = None
            logger.info("Cleaned up ingestion resources")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")