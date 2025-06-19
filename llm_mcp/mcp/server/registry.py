"""
MCP Tool Registry for dynamic tool registration and discovery.

This module handles the registration of framework tools with FastMCP servers,
providing dynamic tool discovery and registration management.
"""

import logging
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

from ...tools.filesystem.search import SearchInFilesTool
from ...tools.filesystem.file_ops import FileReadTool, FileWriteTool, ListDirectoryTool
from ...tools.semantic.search import WorkingSemanticSearchTool
from ...models.semantic import SemanticSearchConfig
from ...utils.constants import DEFAULT_SIMILARITY_THRESHOLD
from ..tools.adapter import MCPToolAdapter

logger = logging.getLogger("llm-mcp-registry")


class MCPToolRegistry:
    """
    Handles tool registration with FastMCP servers.
    
    This class manages the registration of framework tools with MCP servers,
    providing dynamic tool discovery, schema conversion, and registration management.
    """
    
    def __init__(self, mcp_instance: FastMCP, semantic_config: Optional[SemanticSearchConfig] = None):
        """
        Initialize the tool registry.
        
        Args:
            mcp_instance: The FastMCP instance to register tools with
            semantic_config: Configuration for semantic search tool (optional)
        """
        self.mcp = mcp_instance
        self.semantic_config = semantic_config
        self.adapter = MCPToolAdapter()
        self.registered_tools: Dict[str, Any] = {}
        
        # Tool registration mapping - cleaner than if-elif chains
        self.tool_registration_map = {
            "search_in_files": self._register_search_in_files_tool,
            "read_file": self._register_read_file_tool,
            "list_directory": self._register_list_directory_tool,
            "working_semantic_search": self._register_working_semantic_search_tool,
            # "write_file": self._register_write_file_tool,  # DISABLED for security
        }
    
    def register_framework_tools(self) -> None:
        """Register all available framework tools with FastMCP."""
        
        # Get all available framework tools
        tools = self._discover_framework_tools()
        
        # Register each tool
        for tool in tools:
            self._register_single_tool(tool)
        
        logger.info(f"Registered {len(tools)} framework tools with MCP server")
    
    def _discover_framework_tools(self) -> List[Any]:
        """
        Discover all available framework tools.
        
        Returns:
            List of framework tool instances
        """
        # Import and instantiate all available tools
        tools = [
            SearchInFilesTool(),
            FileReadTool(),
            # FileWriteTool(),  # DISABLED: Security risk - see file_ops.py for details
            ListDirectoryTool()
        ]
        
        # Add semantic search tool if configuration is provided and valid
        if self.semantic_config and self._is_semantic_config_valid():
            try:
                semantic_tool = WorkingSemanticSearchTool(config=self.semantic_config)
                tools.append(semantic_tool)
                logger.info("Added WorkingSemanticSearchTool to available tools")
            except Exception as e:
                logger.warning(f"Failed to initialize WorkingSemanticSearchTool: {e}")
        
        return tools
    
    def _is_semantic_config_valid(self) -> bool:
        """
        Validate semantic search configuration.
        
        Returns:
            True if configuration is valid for semantic search
        """
        if not self.semantic_config:
            return False
        
        # Check if qdrant_location is provided and not just the default ":memory:"
        qdrant_location = self.semantic_config.qdrant_config.location
        if not qdrant_location or qdrant_location == ":memory:":
            logger.debug("Semantic search disabled: qdrant_location not configured or using memory")
            return False
        
        # Allow ephemeral databases in /dev/shm or other valid paths
        from pathlib import Path
        db_path = Path(qdrant_location)
        
        # Check if it's an ephemeral database path or if the parent directory exists
        if "/dev/shm" in str(db_path):
            logger.info(f"Semantic search enabled with qdrant_location: {qdrant_location}")
            return True
        elif db_path.exists() or db_path.parent.exists():
            logger.info(f"Semantic search enabled with qdrant_location: {qdrant_location}")
            return True
        else:
            logger.warning(f"Semantic search disabled: qdrant_location path not accessible: {qdrant_location}")
            return False
    
    def _register_single_tool(self, tool: Any) -> None:
        """
        Register a single framework tool with FastMCP.
        
        Args:
            tool: The framework tool to register
        """
        try:
            # Register tool with adapter
            self.adapter.register_tool(tool)
            
            # Create MCP tool function using registration mapping
            registration_func = self.tool_registration_map.get(tool.name)
            if registration_func:
                registration_func(tool)
            else:
                # Generic registration for unknown tools
                self._register_generic_tool(tool)
                logger.info(f"Using generic registration for unknown tool: {tool.name}")
            
            self.registered_tools[tool.name] = tool
            logger.debug(f"Registered tool: {tool.name}")
            
        except Exception as e:
            logger.error(f"Failed to register tool {tool.name}: {e}")
    
    def _register_search_in_files_tool(self, tool: Any) -> None:
        """Register the search_in_files tool."""
        @self.mcp.tool()
        async def search_in_files(
            search_term: str, 
            directory: str = ".", 
            file_extensions: Optional[List[str]] = None, 
            context_lines: int = 2
        ) -> str:
            """Search for a term within files and return matches with context."""
            return await self.adapter.execute_tool(tool.name, {
                "search_term": search_term,
                "directory": directory,
                "file_extensions": file_extensions,
                "context_lines": context_lines
            })
    
    def _register_read_file_tool(self, tool: Any) -> None:
        """Register the read_file tool."""
        @self.mcp.tool()
        async def read_file(files: List[dict], encoding: str = "utf-8") -> str:
            """Read one or multiple files and return their contents.
            
            Args:
                files: List of file specifications, each containing:
                    - file_path (str): Path to the file
                    - focus_lines (Optional[List[int]]): Lines to highlight
                    - line_range (Optional[Tuple[int, int]]): Line range to read
                encoding: File encoding to use
            
            Example:
                files=[{"file_path": "test.py", "focus_lines": [1, 5]}]
                files=[{"file_path": "a.py"}, {"file_path": "b.py", "line_range": [10, 20]}]
            """
            return await self.adapter.execute_tool(tool.name, {
                "files": files,
                "encoding": encoding
            })
    
    def _register_write_file_tool(self, tool: Any) -> None:
        """Register the write_file tool."""
        @self.mcp.tool()
        async def write_file(
            file_path: str, 
            content: str, 
            encoding: str = "utf-8", 
            create_dirs: bool = True
        ) -> str:
            """Write content to a file."""
            return await self.adapter.execute_tool(tool.name, {
                "file_path": file_path,
                "content": content,
                "encoding": encoding,
                "create_dirs": create_dirs
            })
    
    def _register_list_directory_tool(self, tool: Any) -> None:
        """Register the list_directory tool."""
        @self.mcp.tool()
        async def list_directory(
            directory_path: str = ".", 
            show_hidden: bool = False, 
            recursive: bool = False, 
            pattern: Optional[str] = None
        ) -> str:
            """List contents of a directory."""
            return await self.adapter.execute_tool(tool.name, {
                "directory_path": directory_path,
                "show_hidden": show_hidden,
                "recursive": recursive,
                "pattern": pattern
            })
    
    def _register_working_semantic_search_tool(self, tool: Any) -> None:
        """Register the working_semantic_search tool."""
        @self.mcp.tool()
        async def working_semantic_search(
            query: str,
            directory: str = ".",
            max_results: int = 20,
            similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
            search_mode: str = "semantic",
            languages: Optional[List[str]] = None,
            file_patterns: Optional[List[str]] = None,
            exclude_patterns: Optional[List[str]] = None,
            include_context: bool = True,
            context_lines: int = 5,
            reindex: bool = False
        ) -> str:
            """Perform semantic search across code files with proper incremental indexing.
            
            Args:
                query: Natural language search query
                directory: Directory to search in
                max_results: Maximum number of results to return
                similarity_threshold: Minimum similarity score (0.0-1.0)
                search_mode: Search mode (semantic, hybrid, keyword)
                languages: Programming languages to include
                file_patterns: File patterns to include
                exclude_patterns: Patterns to exclude
                include_context: Include surrounding code context
                context_lines: Lines of context around matches
                reindex: Force re-indexing of files
            """
            return await self.adapter.execute_tool(tool.name, {
                "query": query,
                "directory": directory,
                "max_results": max_results,
                "similarity_threshold": similarity_threshold,
                "search_mode": search_mode,
                "languages": languages,
                "file_patterns": file_patterns,
                "exclude_patterns": exclude_patterns,
                "include_context": include_context,
                "context_lines": context_lines,
                "reindex": reindex
            })
    
    def _register_generic_tool(self, tool: Any) -> None:
        """
        Register a generic tool using dynamic function creation.
        
        Args:
            tool: The framework tool to register
        """
        # Create a dynamic MCP tool function
        mcp_function = self.adapter.create_mcp_tool_function(tool)
        
        # Register with MCP
        self.mcp.tool()(mcp_function)
    
    def get_registered_tools(self) -> Dict[str, Any]:
        """Get all registered tools."""
        return self.registered_tools.copy()
    
    def get_tool_names(self) -> List[str]:
        """Get list of registered tool names."""
        return list(self.registered_tools.keys())
    
    def get_tool_count(self) -> int:
        """Get the number of registered tools."""
        return len(self.registered_tools)
    
    def unregister_tool(self, tool_name: str) -> bool:
        """
        Unregister a tool.
        
        Args:
            tool_name: Name of the tool to unregister
            
        Returns:
            True if tool was unregistered, False if not found
        """
        if tool_name in self.registered_tools:
            del self.registered_tools[tool_name]
            logger.debug(f"Unregistered tool: {tool_name}")
            return True
        return False
    
    def clear_all_tools(self) -> None:
        """Clear all registered tools."""
        self.registered_tools.clear()
        self.adapter.clear_tools()
        logger.debug("Cleared all registered tools")