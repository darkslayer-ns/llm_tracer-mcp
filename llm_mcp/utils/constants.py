"""
Centralized constants for the LLM-MCP framework.

This module contains all shared constants used across the framework to avoid duplication
and ensure consistency.
"""

from typing import Dict, List, Set

# =============================================================================
# PROGRAMMING LANGUAGES AND EXTENSIONS
# =============================================================================

# Comprehensive mapping of languages to their file extensions
LANGUAGE_EXTENSIONS: Dict[str, List[str]] = {
    'python': ['.py', '.pyw', '.pyi'],
    'javascript': ['.js', '.mjs', '.jsx'],
    'typescript': ['.ts', '.tsx', '.d.ts'],
    'java': ['.java'],
    'cpp': ['.cpp', '.cxx', '.cc', '.c++'],
    'c': ['.c', '.h'],
    'csharp': ['.cs'],
    'go': ['.go'],
    'rust': ['.rs'],
    'ruby': ['.rb', '.rake'],
    'php': ['.php', '.phtml'],
    'scala': ['.scala', '.sc'],
    'kotlin': ['.kt', '.kts'],
    'elixir': ['.ex', '.exs'],
    'lua': ['.lua'],
    'perl': ['.pl', '.pm'],
    'sql': ['.sql'],
    'markdown': ['.md', '.markdown'],
    'html': ['.html', '.htm'],
    'proto': ['.proto'],
    'latex': ['.tex', '.latex'],
    'cobol': ['.cob', '.cbl'],
    'r': ['.r', '.R'],
    'shell': ['.sh', '.bash', '.zsh'],
    'yaml': ['.yml', '.yaml'],
    'json': ['.json'],
    'xml': ['.xml'],
    'css': ['.css', '.scss', '.sass'],
    'swift': ['.swift'],
    'dart': ['.dart'],
    'vim': ['.vim'],
    'dockerfile': ['Dockerfile', '.dockerfile'],
    'makefile': ['Makefile', '.mk'],
    'config': ['.conf', '.config', '.ini', '.toml']
}

# All supported languages (derived from LANGUAGE_EXTENSIONS)
SUPPORTED_LANGUAGES: Set[str] = set(LANGUAGE_EXTENSIONS.keys())

# All supported file extensions (flattened from LANGUAGE_EXTENSIONS)
ALL_CODE_EXTENSIONS: List[str] = [
    ext for extensions in LANGUAGE_EXTENSIONS.values() 
    for ext in extensions
]

# =============================================================================
# EXCLUSION PATTERNS
# =============================================================================

# Comprehensive list of directories/patterns to exclude from all operations
EXCLUDED_PATTERNS: Set[str] = {
    # Version control
    '.git', '.svn', '.hg', '.bzr',
    
    # Python
    '__pycache__', '.pytest_cache', '.tox', '.coverage', 
    'venv', '.venv', '.env', 'env', 'site-packages',
    '*.pyc', '*.pyo', '*.pyd', '.Python',
    
    # Node.js
    'node_modules', '.npm', '.yarn', 'bower_components',
    
    # Build outputs
    'build', 'dist', '.build', 'target', 'out', 'bin', 'obj',
    
    # IDE/Editor
    '.idea', '.vscode', '.vs', '.eclipse', '.settings',
    '*.swp', '*.swo', '*~', '.DS_Store', 'Thumbs.db',
    
    # Compiled/Binary files
    '*.so', '*.dll', '*.exe', '*.bin', '*.jar', '*.war', '*.ear',
    '*.class', '*.o', '*.a', '*.lib', '*.dylib',
    
    # Minified/Generated files
    '*.min.js', '*.min.css', '*.bundle.js', '*.bundle.css',
    
    # Logs and temporary files
    '*.log', '*.tmp', '*.temp', '*.cache', '*.bak', '*.backup',
    'logs', 'tmp', 'temp',
    
    # Documentation builds
    '_build', 'docs/_build', '.sphinx',
    
    # Package managers
    '.cargo', '.gradle', '.m2', '.ivy2',
    
    # OS specific
    '.Trash', '.Spotlight-V100', '.fseventsd'
}

# =============================================================================
# PROCESSING LIMITS AND DEFAULTS
# =============================================================================

# File processing limits
MAX_FILE_SIZE_BYTES: int = 1024 * 1024  # 1MB
MAX_SEARCH_RESULTS: int = 1000
MAX_SEARCH_FILES: int = 500
MAX_SEARCH_MATCHES: int = 100

# Chunking defaults
DEFAULT_CHUNK_SIZE: int = 1000
DEFAULT_CHUNK_OVERLAP: int = 200
DEFAULT_CONTEXT_LINES: int = 2

# Search defaults
DEFAULT_SIMILARITY_THRESHOLD: float = 0.35
DEFAULT_MAX_RESULTS: int = 20

# =============================================================================
# EMBEDDING MODEL CONFIGURATIONS
# =============================================================================

# Embedding model configurations
EMBEDDING_MODELS = {
    "codebert": {
        "model_path": "microsoft/codebert-base",
        "dimension": 768,
        "max_length": 512
    },
    "unixcoder": {
        "model_path": "microsoft/unixcoder-base", 
        "dimension": 768,
        "max_length": 512
    },
    "minilm": {
        "model_path": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
        "max_length": 256
    },
    "codebert_st": {
        "model_path": "sentence-transformers/all-mpnet-base-v2",
        "dimension": 768,
        "max_length": 384
    }
}

# Default embedding model
DEFAULT_EMBEDDING_MODEL: str = "codebert_st"

# =============================================================================
# DATABASE CONFIGURATIONS
# =============================================================================

# Qdrant database defaults
QDRANT_DEFAULTS = {
    "collection_name": "code_chunks",
    "vector_size": 384,
    "distance_metric": "Cosine",
    "segment_number": 2,
    "max_segment_size": 100000,
    "memmap_threshold": 20000,
    "indexing_threshold": 10000,
    "hnsw_m": 16,
    "hnsw_ef_construct": 100,
    "full_scan_threshold": 10000
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def should_exclude_path(path_parts: List[str]) -> bool:
    """
    Check if a path should be excluded from search operations.
    
    Args:
        path_parts: List of path components (e.g., from Path.parts)
        
    Returns:
        True if the path should be excluded
    """
    return any(part in EXCLUDED_PATTERNS for part in path_parts)


def is_supported_language(language: str) -> bool:
    """
    Check if a programming language is supported.
    
    Args:
        language: Language name to check
        
    Returns:
        True if the language is supported
    """
    return language.lower() in SUPPORTED_LANGUAGES


def get_language_from_extension(file_extension: str) -> str:
    """
    Get the programming language from a file extension.
    
    Args:
        file_extension: File extension (with or without dot)
        
    Returns:
        Language name or 'unknown' if not found
    """
    if not file_extension.startswith('.'):
        file_extension = '.' + file_extension
    
    for language, extensions in LANGUAGE_EXTENSIONS.items():
        if file_extension.lower() in [ext.lower() for ext in extensions]:
            return language
    
    return 'unknown'


def get_extensions_for_language(language: str) -> List[str]:
    """
    Get all file extensions for a programming language.
    
    Args:
        language: Programming language name
        
    Returns:
        List of file extensions (with dots)
    """
    return LANGUAGE_EXTENSIONS.get(language.lower(), [])


def get_primary_extension_for_language(language: str) -> str:
    """
    Get the primary file extension for a programming language.
    
    Args:
        language: Programming language name
        
    Returns:
        Primary file extension (with dot) or empty string if unknown
    """
    extensions = get_extensions_for_language(language)
    return extensions[0] if extensions else ''


def get_exclude_patterns_list() -> List[str]:
    """
    Get exclusion patterns as a list (for compatibility with existing code).
    
    Returns:
        List of exclusion patterns
    """
    return list(EXCLUDED_PATTERNS)