"""
Built-in tools for the generic LLM-MCP framework.
"""

from .base import BaseTool
from .filesystem import FileReadTool, FileWriteTool, ListDirectoryTool, SearchFilesTool, SearchInFilesTool

# Try to import semantic tools from the semantic folder
try:
    from .semantic import WorkingSemanticSearchTool, SemanticIngestionTool, SemanticQueryTool
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False

__all__ = [
    "BaseTool",
    "FileReadTool",
    "FileWriteTool",
    "ListDirectoryTool",
    "SearchFilesTool",
    "SearchInFilesTool"
]

# Add semantic tools if available
if SEMANTIC_AVAILABLE:
    __all__.extend(["WorkingSemanticSearchTool", "SemanticIngestionTool", "SemanticQueryTool"])