"""
Generic LLM-MCP Framework

A completely generic Python framework for building any LLM-powered application
with Model Context Protocol (MCP) tool integration. The framework is domain-agnostic
and becomes whatever you need based on the prompts and tools you provide.
"""

# Main analyzer function - the primary interface
from .analyzer import analyze_repository_with_llm

# Core framework components
from .llm.session import LLMSession
from .models.requests import GenericRequest
from .models.responses import GenericResponse
from .models.streaming import StreamEvent
from .models.errors import LLMError, ToolError, SessionError, LLMException, ToolException, SessionException

# New clean structure
from .llm.client import LLMClient
from .mcp.client import FastMCPClient
from .mcp.manager import MCPManager
from .mcp.server.server import FastMCPServer

# Backward compatibility imports with deprecation warnings
import warnings

def _deprecated_import(old_module, new_module, name):
    """Helper to show deprecation warnings for old imports."""
    warnings.warn(
        f"Importing {name} from {old_module} is deprecated. "
        f"Use 'from {new_module} import {name}' instead.",
        DeprecationWarning,
        stacklevel=3
    )

# Backward compatibility for old imports
try:
    from .mcp_client import FastMCPClient as _OldFastMCPClient, MCPManager as _OldMCPManager
    # These will still work but show deprecation warnings
except ImportError:
    pass  # Old files might not exist anymore

try:
    from .server.mcp_server import FastMCPServer as _OldFastMCPServer
    # This will still work but show deprecation warnings
except ImportError:
    pass  # Old files might not exist anymore

__version__ = "0.1.0"
__all__ = [
    # Primary interface
    "analyze_repository_with_llm",
    
    # Core framework components
    "LLMSession",
    "LLMClient",
    "GenericRequest",
    "GenericResponse",
    "StreamEvent",
    "LLMError",
    "ToolError",
    "SessionError",
    "LLMException",
    "ToolException",
    "SessionException",
    "FastMCPClient",
    "MCPManager",
    "FastMCPServer"
]